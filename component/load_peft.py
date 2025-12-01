import os
from loguru import logger
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AddedToken,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


def load_tokenizer(args):
    model_path = f"model/{args.model_name}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"[Tokenizer] vocab size: {tokenizer.vocab_size}")
    return tokenizer


def load_model(args, training_args):

    assert training_args.bf16 or training_args.fp16, \
        "You must enable bf16 or fp16."

    logger.info(f"[Model] Loading base model: {args.model_name}")

    model_path = f"model/{args.model_name}"

    # =======================
    #   Base Model Loading
    # =======================
    if args.train_mode == "full":
        logger.info("[Mode] Full-parameter training")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None if training_args.deepspeed else "auto",
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            trust_remote_code=True,
        )
        model = model.to("cuda")

        logger.info("[Model] Full training model loaded.")
        return model

    # =======================
    #   LoRA Training Path
    # =======================
    elif args.train_mode == "lora":
        logger.info("[Mode] LoRA fine-tuning")
        torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None if training_args.deepspeed else "auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            # quantization_config=quantization_config
        )

        model.config.use_cache = False

        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
        ]

        # ========================
        # LoRA Config
        # ========================
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        try:
            import peft.tuners.lora.awq
            peft.tuners.lora.awq.dispatch_awq = lambda *_, **__: None
        except:
            pass

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if training_args.deepspeed:
            model = model.to("cuda")

        logger.info("[Model] LoRA model loaded.")
        return model

    else:
        raise ValueError("train_mode must be one of ['full', 'lora'].")

