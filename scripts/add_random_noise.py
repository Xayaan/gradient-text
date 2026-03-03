#!/usr/bin/env python3

import argparse
import os
import shutil

try:
    import torch
except Exception:
    torch = None

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig


def _copy_base_snapshot(model_path: str, save_folder: str) -> bool:
    if not os.path.isdir(model_path):
        return False
    os.makedirs(save_folder, exist_ok=True)

    ignored_names = {"blobs", "refs", "snapshots", ".locks"}
    copied_any = False
    for name in os.listdir(model_path):
        if name in ignored_names:
            continue
        source = os.path.join(model_path, name)
        destination = os.path.join(save_folder, name)
        try:
            if os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
            copied_any = True
        except Exception as exc:
            print(f"Failed to copy fallback asset {source}: {exc}", flush=True)
    return copied_any


def create_noisy_model(model_path: str, save_folder: str, noise_std: float = 0.01) -> bool:
    if torch is None:
        print("Torch unavailable in add_random_noise; using base snapshot copy fallback.", flush=True)
        return _copy_base_snapshot(model_path, save_folder)

    try:
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

        try:
            model.generation_config = GenerationConfig.from_model_config(model.config)
        except Exception:
            pass

        with torch.no_grad():
            embeddings = model.get_input_embeddings()
            if embeddings is not None and getattr(embeddings, "weight", None) is not None:
                noise = torch.randn_like(embeddings.weight) * noise_std
                embeddings.weight.add_(noise)

        os.makedirs(save_folder, exist_ok=True)
        model.save_pretrained(save_folder)
        tokenizer.save_pretrained(save_folder)
        return True
    except Exception as exc:
        print(f"Failed to create noisy fallback model: {exc}", flush=True)
        return _copy_base_snapshot(model_path, save_folder)


def main():
    parser = argparse.ArgumentParser(description="Create a minimal noisy fallback model.")
    parser.add_argument("model_path", help="Local base model path")
    parser.add_argument("save_folder", help="Output folder")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Noise standard deviation")
    args = parser.parse_args()

    success = create_noisy_model(args.model_path, args.save_folder, noise_std=args.noise_std)
    if not success:
        raise SystemExit("Failed to produce fallback model artifacts.")


if __name__ == "__main__":
    main()
