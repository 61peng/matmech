import argparse
from loguru import logger
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
import json
import torch


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='config/debug_config.json', help="")
    parser.add_argument("--time", type=str, help="", default=None)
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # Read the parameter configuration for training
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    train_args = json.load(open(train_args_file, 'r'))
    args, training_args = parser.parse_dict(train_args)
    
    training_args.output_dir = join(training_args.output_dir, args.train_mode, args.model_name, args.dataset)
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))

   
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    
    set_seed(training_args.seed)
    return args, training_args


def load_sft_dataset(args, tokenizer):
    train_file = f'training_data/{args.dataset}/train.jsonl'
    template = template_dict[args.template_name]    

    train_dataset = UnifiedSFTDataset(train_file, tokenizer, args.max_seq_length, template)
    
    return train_dataset


def init_components(args, training_args):
    """
    init components: tokenizer, dataset, model, trainer
    """
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # load tokenizer
    tokenizer = load_tokenizer(args)

    logger.info('Train model with sft task')
    args.train_dataset = load_sft_dataset(args, tokenizer)#system + user + assistant; target_mask:{0:system + user, 1 : assistant}; {input_ids:tensor,attention_mask,target_mask}
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    
    random.shuffle(args.train_dataset.data_list)
    # load model
    model = load_model(args, training_args)
    # _prepare_model_for_training(model)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    # init Trainer
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