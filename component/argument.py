from dataclasses import dataclass, field
from typing import Optional
from component.dataset import UnifiedSFTDataset

@dataclass
class CustomizedArguments:
    """
    Some custom arguments
    """
    max_seq_length: int = field(metadata={"help": "Maximum input length"})
    model_name: str = field(metadata={"help": "Path to pre-trained weights"})
    dataset: str = field(default="", metadata={"help": "Dataset. If task_type=pretrain, specify a folder to scan all jsonl files under it"})
    train_file: str = field(default="", metadata={"help": "Training dataset. If task_type=pretrain, specify a folder to scan all jsonl files under it"})
    do_prediction: bool = field(default=False, metadata={"help": "Whether to run prediction"})
    template_name: str = field(default="", metadata={"help": "Data format for sft"})
    eval_file: Optional[str] = field(default="", metadata={"help": "Evaluation dataset"})
    tokenize_num_workers: int = field(default=10, metadata={"help": "Number of workers for tokenization during pretraining"})
    task_type: str = field(default="sft", metadata={"help": "Task type: [pretrain, sft]"})
    train_mode: str = field(default="qlora", metadata={"help": "Training mode: [full, qlora]"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    num_virtual_tokens: Optional[int] = field(default=20, metadata={"help": "Number of virtual tokens"})
    train_dataset: UnifiedSFTDataset = field(default=None, metadata={"help": "Training dataset input"})
    test_dataset: UnifiedSFTDataset = field(default=None, metadata={"help": "Testing dataset input"})
