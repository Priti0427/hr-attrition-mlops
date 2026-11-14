"""Run every experiment listed in configs/experiments.yaml and log them to MLflow."""

import argparse
import copy

import yaml

from src.config import load_config
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    args = parser.parse_args()

    base = load_config(args.config)
    with open(args.experiments) as f:
        experiments = yaml.safe_load(f)

    for exp in experiments:
        cfg = copy.deepcopy(base)
        cfg["model"] = exp["model"]
        cfg["data"]["drop_columns"] = cfg["data"]["drop_columns"] + exp.get("drop_extra", [])

        _, val, _, failures = train(cfg, run_name=exp["name"], save_outputs=False)
        status = "ok" if not failures else "below threshold"
        print(f"{exp['name']:<22} val_roc_auc={val['roc_auc']:.4f} "
              f"val_f1={val['f1']:.4f} val_recall={val['recall']:.4f}  {status}")


if __name__ == "__main__":
    main()
