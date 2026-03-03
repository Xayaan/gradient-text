#!/usr/bin/env python3
"""Quick static checks for tournament miner guideline compliance."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def check(name: str, ok: bool, detail: str) -> tuple[bool, str]:
    status = "PASS" if ok else "FAIL"
    return ok, f"[{status}] {name}: {detail}"


def main() -> int:
    results: list[tuple[bool, str]] = []

    tuning = read("miner/endpoints/tuning.py")
    text_trainer = read("scripts/text_trainer.py")
    config_handler = read("core/config/config_handler.py")
    text_dockerfile = read("dockerfiles/standalone-text-trainer.dockerfile")

    results.append(
        check(
            "Training Repo Endpoint",
            '"/training_repo/{task_type}"' in tuning and "get_training_repo" in tuning,
            "Miner exposes required endpoint",
        )
    )
    results.append(
        check(
            "Repo/Commit Config",
            bool(re.search(r'TOURNAMENT_\{task_key\}_REPO', tuning))
            and bool(re.search(r'TOURNAMENT_\{task_key\}_COMMIT', tuning))
            and "TOURNAMENT_REPO" in tuning
            and "TOURNAMENT_COMMIT" in tuning,
            "Per-task and global env overrides available",
        )
    )
    results.append(
        check(
            "Required Text CLI Args",
            all(
                flag in text_trainer
                for flag in [
                    "--task-id",
                    "--model",
                    "--dataset",
                    "--dataset-type",
                    "--task-type",
                    "--expected-repo-name",
                    "--hours-to-complete",
                ]
            ),
            "All standardized text args handled",
        )
    )
    results.append(
        check(
            "Output Path Policy",
            "get_checkpoints_output_path" in text_trainer,
            "Uses trainer path utilities for expected output location",
        )
    )
    results.append(
        check(
            "Offline WandB Logging",
            '"wandb_mode" = "offline"' not in text_trainer and '"wandb_mode"] = "offline"' in text_trainer,
            "WandB runs are offline",
        )
    )
    results.append(
        check(
            "WandB Symlink Patching",
            "patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)" in text_trainer,
            "Offline logs patched before upload",
        )
    )
    results.append(
        check(
            "GRPO Safe Code-Exec Helper",
            "RESTRICTED_EXECUTION_HELPER" in config_handler and "restricted_execution" in config_handler,
            "Reward-function generator injects restricted execution helper",
        )
    )
    results.append(
        check(
            "RestrictedPython Dependency",
            "RestrictedPython" in text_dockerfile,
            "Text trainer image includes RestrictedPython",
        )
    )
    results.append(
        check(
            "Multi-GPU Launch",
            "--num_processes" in text_trainer and "--multi_gpu" in text_trainer,
            "Trainer launches multi-GPU when available",
        )
    )
    results.append(
        check(
            "OOM/Retry Fallback",
            "adjust_config_for_retry" in text_trainer and "emergency fallback" in text_trainer,
            "Adaptive retries and conservative emergency run implemented",
        )
    )
    results.append(
        check(
            "Checkpoint Selection",
            "retain_best_checkpoint" in text_trainer,
            "Removes weaker checkpoints before upload",
        )
    )

    all_ok = True
    for ok, message in results:
        print(message)
        all_ok = all_ok and ok

    if not all_ok:
        print("\nOne or more compliance checks failed.", file=sys.stderr)
        return 1

    print("\nAll compliance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
