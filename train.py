import argparse
from loguru import logger
import os
from os.path import join
import time
import random
from component.collator import PretrainCollator, SFTDataCollator
from component.argument import CustomizedArguments
from component.template import template_dict
from component.load_peft import load_model, load_tokenizer
from component.dataset import UnifiedSFTDataset
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, concatenate_datasets
import datasets
from itertools import chain
from tqdm import tqdm
import json
from torch import nn
import torch
import datetime

def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/sft/qlora/service_config.json', help="")
    parser.add_argument("--time", type=str, help="", default=None)
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    date = args.time
    # Read the parameter configuration for training
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    train_args = json.load(open(train_args_file, 'r'))
    args, training_args = parser.parse_dict(train_args)
    if date:
        training_args.output_dir = join(training_args.output_dir, args.train_mode, args.model_name, date)
    else:
        training_args.output_dir = join(training_args.output_dir, args.train_mode, args.model_name, args.dataset)
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))

   
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    
    set_seed(training_args.seed)
    return args, training_args

def load_pretrain_dataset(training_args, args, tokenizer):
    """
    Multi threaded preprocessing of pre training data
    """
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = args.train_file
    max_seq_length = args.max_seq_length
    # Create cache path
    cache_dir = join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    # Scan all JSON files
    logger.info('Scanning all the training file...')
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    # Preprocess all text, ID it, and perform packing operations
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []  
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file)
            file_name = file_name.replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = join(cache_path, 'tmp')   
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset("json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False)
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
                

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert pretrain_dataset.features.type == processed_dataset["train"].features.type
                pretrain_dataset = concatenate_datasets([pretrain_dataset, processed_dataset["train"]])
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset

def load_sft_dataset(args, tokenizer):
    train_file = f'data/instruct_data/{args.dataset}/train.jsonl'
    template = template_dict[args.template_name]    

    train_dataset = UnifiedSFTDataset(train_file, tokenizer, args.max_seq_length, template)
    
    return train_dataset

def _prepare_model_for_training(model: nn.Module):
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

def init_components(args, training_args):
    """
    初始化各个组件
    """
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # 加载tokenizer
    tokenizer = load_tokenizer(args)

    # 初始化dataset和collator
    if args.task_type == 'pretrain':
        logger.info('Train model with pretrain task')
        args.train_dataset = load_pretrain_dataset(training_args, args, tokenizer)
        data_collator = PretrainCollator(tokenizer, args.max_seq_length)
    else:
        logger.info('Train model with sft task')
        args.train_dataset = load_sft_dataset(args, tokenizer)#system + user + assistant; target_mask:{0:system + user, 1 : assistant}; {input_ids:tensor,attention_mask,target_mask}
        data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    random.shuffle(args.train_dataset.data_list)
    # 加载model
    model = load_model(args, training_args)
    # _prepare_model_for_training(model)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=args.train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer

def main():
    # Perform some configuration and checks
    args, training_args = setup_everything()
    # Load components
    trainer = init_components(args, training_args)
    # start training
    logger.info("*** starting training ***")
    # record start time
    start_time = time.time()
    train_result = trainer.train()
    # record end time
    end_time = time.time()
    # calculate training time
    training_time = end_time - start_time
    # convert to hours, minutes, seconds
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # save model
    final_save_path = join(training_args.output_dir)
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # save training time and max memory
    logger.info(f'max memory usage: {round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)} G')
    logger.info(f"training time {int(hours)}:{int(minutes)}:{int(seconds)}")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()