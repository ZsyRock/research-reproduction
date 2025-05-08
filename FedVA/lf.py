import sys
from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import Untargeted
from federated_learning.utils import replace_0_with_9_1_with_3

from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
from loguru import logger # 使用 loguru 记录日志，输出到标准输出
import os

# 自动查找下一个未被占用的实验编号（避免覆盖现有日志和模型文件）
def find_next_exp_idx(base=3000):
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base

'''
原始运行方式（保留对照）
if __name__ == '__main__':
    START_EXP_IDX = 3000
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 20
    REPLACEMENT_METHOD = replace_0_with_9_1_with_3
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 50
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)
'''

if __name__ == '__main__':
    START_EXP_IDX = find_next_exp_idx()  # 自动确定编号
    NUM_EXP = 1                          # 设置你要运行几个实验
    NUM_POISONED_WORKERS = 0           # 投毒客户端数量
    REPLACEMENT_METHOD = replace_0_with_9_1_with_3  # 替换策略（投毒类型）
    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 50     # 每轮选多少个客户端训练
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)

logger.remove()  # 移除默认 handler
logger.add(sys.stdout, level="INFO") 