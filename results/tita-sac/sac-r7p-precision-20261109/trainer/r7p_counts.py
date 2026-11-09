"""R6任务计数快照；不导入Torch、不改变SAC计算或共享restore。"""
from dataclasses import asdict,is_dataclass

def counts(trainer):
    """父Adam步数另记；返回本任务收集/更新计数的独立字典。"""
    c=trainer.counters
    return asdict(c) if is_dataclass(c) else dict(vars(c))
