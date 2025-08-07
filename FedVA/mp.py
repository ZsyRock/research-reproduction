import sys
import os
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
from loguru import logger

# 自动查找下一个未被占用的实验编号（避免覆盖现有日志和模型文件）
def find_next_exp_idx(base=3000):
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base

# ======= 修改参数入口：支持 noniid 分布与 Dirichlet α =======
def run_model_poison_exp(num_poisoned_workers, kwargs, strategy, experiment_id):
    def config_modifier(args):
        args.model_poison = 'sign'      # 启用 sign-flipping 攻击
        args.sign_scale = -1             # 攻击强度（可调）
        args.data_poison = False        # 关闭数据投毒
        args.mal_strat = None             # 清除数据投毒策略
        args.defence = "MI"             # 启用 Mutual Information 防御（伪逻辑）

        args.data_distribution_strategy = "noniid"   # 开启 Non-IID
        args.noniid_alpha = 0.5                      # 设置 Dirichlet alpha

        return args

    run_exp(
        replacement_method=None,
        num_poisoned_workers=num_poisoned_workers,
        KWARGS=kwargs,
        client_selection_strategy=strategy,
        idx=experiment_id,
        config_modifier=config_modifier
    )

# ======= 实验启动入口 =======
if __name__ == '__main__':
    START_EXP_IDX = find_next_exp_idx()
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 10
    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 50
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_model_poison_exp(NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)

logger.remove()
logger.add(sys.stdout, level="INFO")
