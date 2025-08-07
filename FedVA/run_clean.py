import sys
import os
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
from loguru import logger

def find_next_exp_idx(base=6000):  # 用不同编号区分实验
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base

def config_modifier(args):
    args.model_poison = None
    args.data_poison = False
    args.mal_strat = None
    args.defence = None
    args.data_distribution_strategy = "iid"  # ✅ 这是关键！
    return args

if __name__ == '__main__':
    START_EXP_IDX = find_next_exp_idx()
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 0
    REPLACEMENT_METHOD = None

    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 50
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(
            replacement_method=REPLACEMENT_METHOD,
            num_poisoned_workers=NUM_POISONED_WORKERS,
            KWARGS=KWARGS,
            client_selection_strategy=RandomSelectionStrategy(),
            idx=experiment_id,
            config_modifier=config_modifier
        )

logger.remove()
logger.add(sys.stdout, level="INFO")
