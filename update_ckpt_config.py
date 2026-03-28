#!/usr/bin/env python3
"""
If fine-tuned checkpoints are mounted at /app/ckpts/finetuned/, update the
ensemble config to use them instead of the release checkpoints.
"""
import os
import sys

CONFIG_PATH = "/app/configs/ensemble_2split_sampling.yaml"
FINETUNED_DIR = "/app/ckpts/finetuned"
CKPT_1 = "A2SB_twosplit_0.0_0.5_finetuned.ckpt"
CKPT_2 = "A2SB_twosplit_0.5_1.0_finetuned.ckpt"


def main() -> int:
    path1 = os.path.join(FINETUNED_DIR, CKPT_1)
    path2 = os.path.join(FINETUNED_DIR, CKPT_2)
    if not (os.path.isfile(path1) and os.path.isfile(path2)):
        return 0

    try:
        import yaml
    except ImportError:
        return 0

    if not os.path.isfile(CONFIG_PATH):
        return 0

    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)

    if "model" not in data or "pretrained_checkpoints" not in data["model"]:
        return 0

    data["model"]["pretrained_checkpoints"] = [path1, path2]
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print("[update_ckpt_config] Using fine-tuned checkpoints", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
