# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os

# # If there is Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
# os.environ["MKL_THREADING_LAYER"] = "GNU"
# import numpy as np
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import numpy as np 
import json
import argparse
import glob
import subprocess
import yaml
import time 
from datetime import datetime
import shutil
import csv
from tqdm import tqdm

import librosa
import soundfile as sf


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def save_yaml(data, prefix="../configs/temp"):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_num = np.random.rand()
    rnd_num = rnd_num - rnd_num % 0.000001
    file_name = f"{prefix}_{timestamp}_{rnd_num}.yaml"
    with open(file_name, 'w') as f:
        yaml.dump(data, f)
    return file_name


def shell_run_cmd(cmd, cwd=None):
    print('running:', " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        check=False,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"stderr:\n{result.stderr}"
        )


def compute_rolloff_freq(audio_file, roll_percent=0.99):
    """Fallback if no explicit cutoff is provided."""
    y, sr = librosa.load(audio_file, sr=None)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    rolloff = int(np.mean(rolloff))
    print('Auto-detected 99 percent rolloff:', rolloff)
    return rolloff


def upsample_one_sample(
    audio_filename,
    output_audio_filename,
    predict_n_steps=50,
    explicit_cutoff=None,
    predict_batch_size=16,
):

    assert output_audio_filename != audio_filename, "output filename cannot be input filename"

    inference_config = load_yaml('../configs/inference_files_upsampling.yaml')
    inference_config['data']['predict_filelist'] = [{
        'filepath': audio_filename,
        'output_subdir': '.'
    }]

    if explicit_cutoff is not None and explicit_cutoff > 0:
        print(f"Using explicit cutoff frequency: {explicit_cutoff} Hz")
        cutoff_freq = int(explicit_cutoff)
    else:
        print("No explicit cutoff provided, attempting to auto-detect...")
        cutoff_freq = compute_rolloff_freq(audio_filename, roll_percent=0.99)

    mask_config = {
        'min_cutoff_freq': cutoff_freq,
        'max_cutoff_freq': cutoff_freq
    }

    base_transform_list = inference_config['data'].get('transforms_aug', [])

    if not base_transform_list:
        print("WARNING: transforms_aug is empty in base config! Masking may fail.")
    else:
        base_transform_list[0]['init_args']['upsample_mask_kwargs'] = mask_config

    # Mirror the upsample mask across all transform lists so predict mode picks
    # up the explicit cutoff consistently.
    print(f"Applying mask (Cutoff: {cutoff_freq}Hz) to all transform lists...")
    inference_config['data']['transforms_aug'] = base_transform_list
    inference_config['data']['transforms_aug_val'] = base_transform_list
    inference_config['data']['eval_transforms_aug'] = base_transform_list
    temporary_yaml_file = save_yaml(inference_config)

    cmd = [
        "python",
        "ensembled_inference_api.py",
        "predict",
        "-c", "configs/ensemble_2split_sampling.yaml",
        "-c", temporary_yaml_file.replace('../', ''),
        f"--model.predict_n_steps={predict_n_steps}",
        f"--model.predict_batch_size={predict_batch_size}",
        f"--model.output_audio_filename={output_audio_filename}",
    ]
    shell_run_cmd(cmd, cwd="../")

    if os.path.exists(temporary_yaml_file):
        os.remove(temporary_yaml_file)


def main():
    parser = argparse.ArgumentParser(description='A2SB Upsampler API')
    parser.add_argument('-f','--audio_filename', type=str, help='audio filename to be upsampled', required=True)
    parser.add_argument('-o','--output_audio_filename', type=str, help='path to save upsampled audio', required=True)
    parser.add_argument('-n','--predict_n_steps', type=int, help='number of sampling steps', default=50)
    parser.add_argument('-c','--cutoff', type=float, help='Explicit cutoff frequency in Hz', default=None)
    parser.add_argument('-b','--batch_size', type=int, help='Predict batch size', default=16)
    args = parser.parse_args()

    upsample_one_sample(
        audio_filename=args.audio_filename,
        output_audio_filename=args.output_audio_filename,
        predict_n_steps=args.predict_n_steps,
        explicit_cutoff=args.cutoff,
        predict_batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()

    # python A2SB_upsample_api.py -f <INPUT_FILENAME> -o <OUTPUT_FILENAME> -n <N_STEPS>
