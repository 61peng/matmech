from dataclasses import dataclass, field
from typing import Optional
from component.dataset import UnifiedSFTDataset

@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    model_name: str = field(metadata={"help": "预训练权重路径"})
    dataset: str = field(default="", metadata={"help": "数据集。如果task_type=pretrain，请指定文件夹，将扫描其下面的所有jsonl文件"})
    train_file: str = field(default="", metadata={"help": "训练集。如果task_type=pretrain，请指定文件夹，将扫描其下面的所有jsonl文件"})
    do_prediction: bool = field(default=False, metadata={"help": "是否进行预测"})
    template_name: str = field(default="", metadata={"help": "sft时的数据格式"})
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    tokenize_num_workers: int = field(default=10, metadata={"help": "预训练时tokenize的线程数量"})
    task_type: str = field(default="sft", metadata={"help": "预训练任务：[pretrain, sft]"})
    train_mode: str = field(default="qlora", metadata={"help": "训练方式：[full, qlora]"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    num_virtual_tokens: Optional[int] = field(default=20, metadata={"help": "number of virtual tokens"})
    train_dataset: UnifiedSFTDataset = field(default=None, metadata={"help": "输入数据"})
    test_dataset: UnifiedSFTDataset = field(default=None, metadata={"help": "输入数据"})
