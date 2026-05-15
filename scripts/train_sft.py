"""
train_sft.py — SymLoRA SFT 학습 스크립트

사용법:
    # standard init (baseline)
    python scripts/train_sft.py data.task=math

    # symmetric init
    python scripts/train_sft.py data.task=math init.mode=symmetric

    # pissa init
    python scripts/train_sft.py data.task=math init.mode=pissa
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig

from src.training.sft import SFTRunConfig, train


@hydra.main(config_path="../configs", config_name="train_sft", version_base=None)
def main(cfg: DictConfig):
    data_path = f"data/processed/{cfg.data.task}"

    run_cfg = SFTRunConfig(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=list(cfg.model.lora_target_modules),
        data_path=data_path,
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        max_grad_norm=cfg.training.max_grad_norm,
        optim=cfg.training.optim,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        mlflow_tracking_uri=cfg.mlflow.tracking_uri,
        mlflow_experiment_name=cfg.mlflow.experiment_name,
        run_name=cfg.training.run_name,
        tau_prev_b_paths=list(cfg.orthogonal.tau_prev_b_paths) if cfg.orthogonal.tau_prev_b_paths else None,
        tau_prev_a_paths=list(cfg.orthogonal.tau_prev_a_paths) if cfg.orthogonal.tau_prev_a_paths else None,
        orth_target=cfg.orthogonal.orth_target,
        orthogonal_alpha=cfg.orthogonal.alpha,
        random_state=cfg.model.random_state,
        init_mode=cfg.init.mode,
        freeze_a=cfg.init.get("freeze_a", False),
        residual_b=cfg.init.get("residual_b", False),
    )

    train(run_cfg)


if __name__ == "__main__":
    main()
