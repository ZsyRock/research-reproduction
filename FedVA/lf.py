import sys
import os

from federated_learning.utils import replace_0_with_9_1_with_3
from federated_learning.worker_selection import RandomSelectionStrategy
from federated_learning.nets.resnet_cifar import ResNet18
from federated_learning.nets.cifar_10_cnn import Cifar10CNN 
from server import run_exp
from loguru import logger


# 自动查找下一个未被占用的实验编号
def find_next_exp_idx(base=5000):
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base


# 设置实验运行逻辑（启用标签翻转攻击 + Non-IID 分布）
def run_model_poison_exp(replacement_method, num_poisoned_workers, kwargs, strategy, experiment_id):
    def config_modifier(args):
        args.batch_size = 8
        args.test_batch_size = 500
        args.lr = 0.01
        args.cuda = True
        args.net = ResNet18
        args.layer_name = "fc.weight"  # corresponds to ResNet18, if using Cifar10CNN, change to "fc2.weight"
        args.model_poison = None              # 不使用模型层面的 sign-flipping 攻击
        args.data_poison = True               # 启用数据投毒（即标签翻转）
        args.mal_strat = "concat"             # 使用 concat 数据注入方式
        args.defence = "PCA"                   # 暂时不使用防御方法（如 MI 或 PCA）

        args.data_distribution_strategy = "noniid"
        args.noniid_alpha = 0.5

        return args

    run_exp(
        replacement_method=replacement_method,
        num_poisoned_workers=num_poisoned_workers,
        KWARGS=kwargs,
        client_selection_strategy=strategy,
        idx=experiment_id,
        config_modifier=config_modifier
    )


# ========= 主入口 =========
if __name__ == '__main__':
    START_EXP_IDX = find_next_exp_idx()
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 20
    REPLACEMENT_METHOD = replace_0_with_9_1_with_3

    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 50
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_model_poison_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)

logger.remove()
logger.add(sys.stdout, level="INFO")
