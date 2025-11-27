import os
from peft import (
    LoraConfig, 
    PromptEncoderConfig, 
    PrefixTuningConfig, 
    PromptTuningConfig, 
    PromptTuningInit, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType)

from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    AddedToken
)
import torch
import bitsandbytes as bnb
import json

def load_tokenizer(args):
    model_path = f'model/{args.model_name}'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
    )
    # 部分模型的base与chat版本的tokenizer存在差异
    if args.template_name == 'internlm2':
        tokenizer._added_tokens_encoder.update({'<|im_start|>': 92543})
        tokenizer._added_tokens_encoder.update({'<|im_end|>': 92542})
        tokenizer._added_tokens_decoder.update({92543: AddedToken('<|im_start|>')})
        tokenizer._added_tokens_decoder.update({92542: AddedToken('<|im_end|>')})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    elif 'orion' in args.model_name.lower():
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    return tokenizer


def find_all_linear_names(model, args):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    if args.train_mode == 'qlora':
        cls = bnb.nn.Linear4bit
    else:
        cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_model(args, training_args):
    """
    加载模型
    """
    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    # 加载模型
    logger.info(f'Loading model from base model: {args.model_name}')
    model_path = f'model/{args.model_name}'
    # 全量训练
    if args.train_mode == 'full':
        logger.info('Training model with full parameters')
        # world_size = int(os.environ.get("WORLD_SIZE", 1))
        # ddp = world_size != 1
        # training_args.ddp_find_unused_parameters = False if ddp else None
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None,
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to('cuda')
        
    # 使用LoRA或者QLoRA训练模型
    elif args.train_mode == 'lora' or args.train_mode == 'qlora':
        # training_args.ddp_find_unused_parameters = False
        if training_args.deepspeed:
            device_map = None
        else:
            device_map = 'auto'
        # todo 适配lora
        if args.train_mode == 'lora':
            logger.info('Training model with LoRA')
            torch_dtype = None
            quantization_config = None
        elif args.train_mode == 'qlora':
            logger.info('Training model with QLoRA')
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,  # 使用嵌套量化来量化已经量化的权重
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            raise Exception('train_mode should be in [full, qlora]')

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,  # 若使用QLoRA，未被量化的层会以fp16的类型加载
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        
        model.config.use_cache = False
        # casts all the non int8 modules to full precision (fp32) for stability
        if args.train_mode == 'qlora':
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        # 找到所有需要插入adapter的全连接层
        target_modules = find_all_linear_names(model, args)
        # 初始化lora配置
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        model.config.torch_dtype = torch.float32


        if training_args.deepspeed:
          model = model.to('cuda')
    

    # moe模型，需要考虑负载均衡的loss
    if 'output_router_logits' in model.config.to_dict():
        logger.info('set output_router_logits as True')
        model.config.output_router_logits = True
    logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return model