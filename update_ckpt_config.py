#!/usr/bin/env python3
"""
If fine-tuned checkpoints are mounted at /app/ckpts/finetuned/, update the
ensemble config to use them instead of the release checkpoints.

The two checkpoints cover different halves of the diffusion trajectory
(t 0.0-0.5 and t 0.5-1.0) and BOTH run on every restoration, so they are
substituted independently: fine-tuning one split and leaving the other on its
release weights is a valid, and common, intermediate state. Whatever is present
is swapped in; anything missing keeps the release checkpoint it already had.
"""
import os
import sys

CONFIG_PATH = "/app/configs/ensemble_2split_sampling.yaml"
FINETUNED_DIR = "/app/ckpts/finetuned"
# index in pretrained_checkpoints -> fine-tuned filename for that split
SPLIT_CKPTS = [
    "A2SB_twosplit_0.0_0.5_finetuned.ckpt",
    "A2SB_twosplit_0.5_1.0_finetuned.ckpt",
]


def main() -> int:
    found = {
        idx: os.path.join(FINETUNED_DIR, name)
        for idx, name in enumerate(SPLIT_CKPTS)
        if os.path.isfile(os.path.join(FINETUNED_DIR, name))
    }
    if not found:
        return 0

    try:
        import yaml
    except ImportError:
        print("[update_ckpt_config] pyyaml missing; keeping release checkpoints",
              file=sys.stderr)
        return 0

    if not os.path.isfile(CONFIG_PATH):
        return 0

    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)

    model = data.get("model") or {}
    ckpts = model.get("pretrained_checkpoints")
    if not isinstance(ckpts, list) or len(ckpts) != len(SPLIT_CKPTS):
        return 0

    ckpts = list(ckpts)
    for idx, path in found.items():
        ckpts[idx] = path
    data["model"]["pretrained_checkpoints"] = ckpts

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    for idx, name in enumerate(SPLIT_CKPTS):
        kind = "fine-tuned" if idx in found else "release"
        print(f"[update_ckpt_config] split {idx}: {kind} -> {ckpts[idx]}",
              file=sys.stderr)
    if len(found) < len(SPLIT_CKPTS):
        missing = [n for i, n in enumerate(SPLIT_CKPTS) if i not in found]
        print(f"[update_ckpt_config] NOTE: no fine-tuned checkpoint for "
              f"{', '.join(missing)}; that half of the diffusion trajectory "
              f"still uses release weights.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
