import os
from loguru import logger
from federated_learning.arguments import Arguments
from defenses.gradient_analysis import extract_gradients
from defenses.mi_clustering import cluster_step1_GM


def find_latest_exp_idx(base=3000, limit=10000, fallback=3000):
    for idx in reversed(range(base, limit)):
        if os.path.exists(f"logs/{idx}.log") and os.path.exists(f"{idx}_models"):
            return idx
    print(f"[WARN] No valid experiment index found. Falling back to default experiment index {fallback}.")
    return fallback


def run_defense_analysis():
    args = Arguments(logger)
    args.log()

    exp_id = find_latest_exp_idx()
    model_path = f"./{exp_id}_models"
    model_files = sorted(os.listdir(model_path))
    logger.info(f"Loaded {len(model_files)} model files from {model_path}")

    layer_name = "fc2.weight"
    class_num = 1
    epochs = list(range(1, 11))
    save_dir = os.path.join("figures", f"GS_{exp_id}")
    os.makedirs(save_dir, exist_ok=True)

    dim_reduced, worker_ids = extract_gradients(args, model_files, model_path, epochs, layer_name, class_num, logger)
    benign_idx, _ = cluster_step1_GM(dim_reduced, np.array(worker_ids), save_dir=save_dir)

    logger.info(f"Defense analysis completed. Benign workers detected: {benign_idx}")
