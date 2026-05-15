"""
SFT 학습 로직
Unsloth + TRL SFTTrainer 기반

OrthogonalSFTTrainer:
  L_total = L_SFT + alpha * mean_layers( ||Q_prev^T @ B_i||_F² / ||B_i||_F² )

  B_i    : 현재 학습 중인 task의 LoRA B 행렬 (d_out, r)
  Q_prev : 이전 K개 task의 B를 concat 후 QR 분해한 orthonormal basis (d_out, ≤K*r)

  기존 방식(각 prev B마다 matmul): K번 연산
  QR 방식: 1번 연산. K가 늘어도 레이어당 matmul 크기 동일.

  의미: loss_layer = ||proj_{span(prev)}(B_i)||² / ||B_i||² ∈ [0, 1]
        0 → 완전 직교 / 1 → 완전히 이전 subspace 내부
"""

from dataclasses import dataclass
from typing import Optional

import mlflow
import torch
from datasets import load_from_disk
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

_UNSLOTH_CACHE = str(Path.home() / ".cache/huggingface/hub/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331")

def _resolve_model(name: str) -> str:
    """unsloth 허브 모델명 → 로컬 캐시 경로"""
    if name == "unsloth/Llama-3.2-3B":
        return _UNSLOTH_CACHE
    return name


@dataclass
class SFTRunConfig:
    # 모델
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    # 데이터
    data_path: str
    # 학습
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float   # warmup_steps 계산용 (SFTConfig엔 steps로 변환해서 전달)
    max_grad_norm: float
    optim: str
    bf16: bool
    logging_steps: int
    save_strategy: str
    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    run_name: str
    # Orthogonal Loss — 여러 이전 task를 동시에 직교화 가능
    tau_prev_b_paths: Optional[list[str]] = None   # 이전 task들의 B 행렬 경로 list
    tau_prev_a_paths: Optional[list[str]] = None   # 이전 task들의 A 행렬 경로 list
    orth_target: str = "A"                         # "A" | "B" | "AB"
    orthogonal_alpha: float = 0.0
    random_state: int = 42
    init_mode: str = "standard"                    # "standard" | "symmetric" | "pissa"
    freeze_a: bool = False                         # True → A gradient 차단, B만 학습
    residual_b: bool = False                       # True → LoRA B 모듈 간 residual connection


class OrthogonalSFTTrainer(SFTTrainer):
    """
    SFTTrainer + A/B/AB-subspace Orthogonal Loss (QR projection)

    orth_target에 따라:
      "B": col(B_curr) ⊥ col(B_prev)   — B의 column space 직교화
      "A": row(A_curr) ⊥ row(A_prev)   — A의 row space 직교화
      "AB": 둘 다

    QR 분해로 이전 task들의 subspace를 orthonormal basis로 압축.
    loss_layer = ||Q_prev^T @ M||_F² / (||M||_F² + eps)
      B-orth: M = B_i (d_out, r),  Q_prev from prev B columns
      A-orth: M = A_i^T (d_in, r), Q_prev from prev A rows (= A^T columns)
    """

    def __init__(
        self,
        *args,
        tau_prev_b_paths: Optional[list[str]] = None,
        tau_prev_a_paths: Optional[list[str]] = None,
        orth_target: str = "A",
        alpha: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.orth_target = orth_target.upper()
        self.Q_prev_B: Optional[dict[str, torch.Tensor]] = None
        self.Q_prev_A: Optional[dict[str, torch.Tensor]] = None

        if alpha > 0.0:
            if self.orth_target in ("B", "AB") and tau_prev_b_paths:
                self.Q_prev_B = self._build_Q_prev_B(tau_prev_b_paths)
                n_params = sum(q.numel() for q in self.Q_prev_B.values())
                print(f"[OrthSFT] B-orth: {len(self.Q_prev_B)} layers, {n_params/1e6:.2f}M params")

            if self.orth_target in ("A", "AB") and tau_prev_a_paths:
                self.Q_prev_A = self._build_Q_prev_A(tau_prev_a_paths)
                n_params = sum(q.numel() for q in self.Q_prev_A.values())
                print(f"[OrthSFT] A-orth: {len(self.Q_prev_A)} layers, {n_params/1e6:.2f}M params")

            print(f"[OrthSFT] target={self.orth_target}, alpha={alpha}")

    @staticmethod
    def _build_Q_prev_B(paths: list[str]) -> dict[str, torch.Tensor]:
        """B 행렬들을 concat 후 QR → orthonormal basis (column space)"""
        layer_mats: dict[str, list[torch.Tensor]] = {}
        for path in paths:
            tau = torch.load(path, map_location="cpu", weights_only=True)
            for name, B in tau.items():
                layer_mats.setdefault(name, []).append(B.float())

        Q_prev = {}
        for name, mats in layer_mats.items():
            cat = torch.cat(mats, dim=1)              # (d_out, K*r)
            Q, _ = torch.linalg.qr(cat, mode="reduced")
            Q_prev[name] = Q
        return Q_prev

    @staticmethod
    def _build_Q_prev_A(paths: list[str]) -> dict[str, torch.Tensor]:
        """A 행렬들의 row space를 QR → orthonormal basis. A^T의 column space."""
        layer_mats: dict[str, list[torch.Tensor]] = {}
        for path in paths:
            tau = torch.load(path, map_location="cpu", weights_only=True)
            for name, A in tau.items():
                # A: (r, d_in) → A^T: (d_in, r)
                layer_mats.setdefault(name, []).append(A.T.float())

        Q_prev = {}
        for name, mats in layer_mats.items():
            cat = torch.cat(mats, dim=1)              # (d_in, K*r)
            Q, _ = torch.linalg.qr(cat, mode="reduced")
            Q_prev[name] = Q
        return Q_prev

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if self.alpha <= 0.0 or (self.Q_prev_B is None and self.Q_prev_A is None):
            return result

        if return_outputs:
            loss, outputs = result
        else:
            loss = result

        orth_loss_b = torch.tensor(0.0, device=loss.device)
        orth_loss_a = torch.tensor(0.0, device=loss.device)

        if self.Q_prev_B is not None:
            orth_loss_b = self._orth_loss_B(model)
        if self.Q_prev_A is not None:
            orth_loss_a = self._orth_loss_A(model)

        orth_loss = orth_loss_b + orth_loss_a
        loss = loss + self.alpha * orth_loss

        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {"orth_loss": orth_loss.item()}
            if self.Q_prev_B is not None:
                log_dict["orth_loss_B"] = orth_loss_b.item()
            if self.Q_prev_A is not None:
                log_dict["orth_loss_A"] = orth_loss_a.item()
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def _orth_loss_B(self, model) -> torch.Tensor:
        """col(B_curr) ⊥ col(B_prev)"""
        device = next(model.parameters()).device
        total = torch.tensor(0.0, device=device)
        count = 0

        for name, module in model.named_modules():
            if not hasattr(module, "lora_B") or name not in self.Q_prev_B:
                continue
            adapter_key = list(module.lora_B.keys())[0]
            B_i = module.lora_B[adapter_key].weight              # (d_out, r)
            Q = self.Q_prev_B[name].to(device=device, dtype=B_i.dtype)

            proj = Q.T @ B_i                                     # (rank_Q, r)
            total = total + (proj ** 2).sum() / ((B_i ** 2).sum() + 1e-8)
            count += 1

        return total / max(count, 1)

    def _orth_loss_A(self, model) -> torch.Tensor:
        """row(A_curr) ⊥ row(A_prev) — A^T의 column space 기준"""
        device = next(model.parameters()).device
        total = torch.tensor(0.0, device=device)
        count = 0

        for name, module in model.named_modules():
            if not hasattr(module, "lora_A") or name not in self.Q_prev_A:
                continue
            adapter_key = list(module.lora_A.keys())[0]
            A_i = module.lora_A[adapter_key].weight              # (r, d_in)
            A_i_T = A_i.T                                        # (d_in, r)
            Q = self.Q_prev_A[name].to(device=device, dtype=A_i.dtype)

            proj = Q.T @ A_i_T                                   # (rank_Q, r)
            total = total + (proj ** 2).sum() / ((A_i_T ** 2).sum() + 1e-8)
            count += 1

        return total / max(count, 1)


@torch.no_grad()
def _log_lora_norms(model, tag: str = "init"):
    """각 LoRA 모듈의 ‖A‖, ‖B‖ Frobenius norm 통계를 출력/로깅.

    Adam은 step size를 grad magnitude로 normalize하므로, 파라미터 절대 크기가
    크면 relative step(‖Δθ‖ / ‖θ‖)이 작아진다. PiSSA처럼 ‖B‖ ≈ top singular
    values 스케일이면 학습 중 B가 거의 안 움직일 수 있다.
    """
    import statistics
    a_norms, b_norms, ratios = [], [], []
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or len(module.lora_A) == 0:
            continue
        adapter_key = next(iter(module.lora_A.keys()))
        A = module.lora_A[adapter_key].weight
        B = module.lora_B[adapter_key].weight
        a_n = A.detach().float().norm().item()
        b_n = B.detach().float().norm().item()
        a_norms.append(a_n)
        b_norms.append(b_n)
        ratios.append(b_n / (a_n + 1e-12))

    if not a_norms:
        return {}

    stats = {
        f"{tag}/A_norm_mean": statistics.mean(a_norms),
        f"{tag}/A_norm_median": statistics.median(a_norms),
        f"{tag}/B_norm_mean": statistics.mean(b_norms),
        f"{tag}/B_norm_median": statistics.median(b_norms),
        f"{tag}/BA_ratio_mean": statistics.mean(ratios),
        f"{tag}/BA_ratio_median": statistics.median(ratios),
        f"{tag}/n_layers": len(a_norms),
    }
    print(
        f"[lora_norms/{tag}] n={len(a_norms)} | "
        f"‖A‖ mean={stats[f'{tag}/A_norm_mean']:.4f} med={stats[f'{tag}/A_norm_median']:.4f} | "
        f"‖B‖ mean={stats[f'{tag}/B_norm_mean']:.4f} med={stats[f'{tag}/B_norm_median']:.4f} | "
        f"‖B‖/‖A‖ mean={stats[f'{tag}/BA_ratio_mean']:.4f} med={stats[f'{tag}/BA_ratio_median']:.4f}"
    )
    return stats


@torch.no_grad()
def _capture_base_weight_sample(model):
    """첫 번째 LoRA 레이어의 이름과 base weight 스냅샷 반환 (init 전에 호출)."""
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or len(module.lora_A) == 0:
            continue
        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        return name, base_layer.weight.detach().float().clone()
    return None, None


@torch.no_grad()
def _sanity_check_init(model, init_info, w_before_sample):
    """Base weight가 expected_delta만큼 실제로 수정됐는지 검증.

    검증: W_before - W_after ≈ scaling * B_init @ A_init
    bfloat16 반올림 오차(~1%)를 허용하므로 5% 이내면 OK.
    """
    name_ref, W_before = w_before_sample
    if W_before is None:
        return

    init_weights = init_info.get("init_weights", {})
    if not init_weights:
        return

    for name, module in model.named_modules():
        if name != name_ref:
            continue
        if not hasattr(module, "lora_A") or len(module.lora_A) == 0:
            continue

        a_key = next((k for k in init_weights if k.endswith("lora_A.weight") and name in k), None)
        if a_key is None:
            continue
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in init_weights:
            continue

        adapter_key = next(iter(module.lora_A.keys()))
        scaling = float(module.scaling[adapter_key]) if hasattr(module, "scaling") else 1.0
        device = module.lora_A[adapter_key].weight.device

        A = init_weights[a_key].to(device=device, dtype=torch.float32)
        B = init_weights[b_key].to(device=device, dtype=torch.float32)
        expected_delta = scaling * (B @ A)

        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        W_after = base_layer.weight.detach().float()
        actual_delta = W_before.to(device) - W_after  # 실제로 빠진 양

        rel_err = (actual_delta - expected_delta).norm() / (expected_delta.norm() + 1e-8)
        if rel_err < 0.05:
            print(f"[sanity_check] OK — base weight 보정 반영됨 (상대오차={rel_err:.2e})")
        else:
            print(f"[WARN] sanity check FAILED: base weight 보정 미반영 (상대오차={rel_err:.2e})")
        return


def load_model_and_tokenizer(cfg: SFTRunConfig):
    model_path = _resolve_model(cfg.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _tokenize(ds, tokenizer, max_seq_length: int):
    """
    LoRI 방식 completion-only tokenization.
    prompt 부분 labels = -100 → loss 계산 제외
    completion 부분만 loss 계산
    """
    def tok(batch):
        input_ids_list, labels_list = [], []

        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            # 프롬프트 토큰 길이 계산 (add_special_tokens=True: BOS 포함)
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]

            # 전체 (prompt + completion + EOS) 토큰화
            full_text = prompt + completion
            full = tokenizer(
                full_text,
                add_special_tokens=True,
                max_length=max_seq_length,
                truncation=True,
            )
            full_ids = full["input_ids"]

            # completion 뒤에 EOS 추가 (잘리지 않은 경우에만)
            if len(full_ids) < max_seq_length:
                full_ids = full_ids + [tokenizer.eos_token_id]

            # labels: prompt 부분은 -100, completion 부분만 학습
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            # 길이 맞춤
            full_ids = full_ids[:max_seq_length]
            labels   = labels[:max_seq_length]

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    return ds.map(tok, batched=True, remove_columns=["prompt", "completion"], num_proc=None)


def build_trainer(model, tokenizer, train_ds, eval_ds, cfg: SFTRunConfig) -> SFTTrainer:
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # 메인 프로세스에서 tokenize → _prepare_dataset 건너뜀
    print("Tokenizing datasets...")
    train_ds = _tokenize(train_ds, tokenizer, cfg.max_seq_length)
    eval_ds  = _tokenize(eval_ds,  tokenizer, cfg.max_seq_length)

    # warmup_steps 계산 (warmup_ratio deprecated in transformers 5.x)
    steps_per_epoch = len(train_ds) // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_train_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        max_length=cfg.max_seq_length,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="no",
        save_strategy=cfg.save_strategy,
        report_to="mlflow",
        packing=False,
        gradient_checkpointing=False,  # residual_b의 cross-layer state와 비호환
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=cfg.random_state,
        data_seed=cfg.random_state,
    )

    has_prev = bool(cfg.tau_prev_b_paths) or bool(cfg.tau_prev_a_paths)
    use_orth = cfg.orthogonal_alpha > 0.0 and has_prev
    TrainerClass = OrthogonalSFTTrainer if use_orth else SFTTrainer

    kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    if use_orth:
        kwargs["tau_prev_b_paths"] = list(cfg.tau_prev_b_paths) if cfg.tau_prev_b_paths else None
        kwargs["tau_prev_a_paths"] = list(cfg.tau_prev_a_paths) if cfg.tau_prev_a_paths else None
        kwargs["orth_target"] = cfg.orth_target
        kwargs["alpha"] = cfg.orthogonal_alpha

    trainer = TrainerClass(**kwargs)
    return trainer


def train(cfg: SFTRunConfig):
    import os
    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.random_state)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params({
            "model": cfg.model_name,
            "lora_r": cfg.lora_r,
            "lr": cfg.learning_rate,
            "batch_size": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
            "epochs": cfg.num_train_epochs,
            "data_path": cfg.data_path,
            "orthogonal_alpha": cfg.orthogonal_alpha,
            "orth_target": cfg.orth_target,
            "tau_prev_b_paths": ",".join(cfg.tau_prev_b_paths) if cfg.tau_prev_b_paths else "none",
            "tau_prev_a_paths": ",".join(cfg.tau_prev_a_paths) if cfg.tau_prev_a_paths else "none",
            "init_mode": cfg.init_mode,
            "random_state": cfg.random_state,
        })

        print("모델 로드 중...")
        model, tokenizer = load_model_and_tokenizer(cfg)

        if cfg.init_mode != "standard":
            from src.models.sym_lora import apply_symmetric_init
            from safetensors.torch import save_file as _save_safetensors

            w_before = _capture_base_weight_sample(model)  # init 전 스냅샷

            init_info = apply_symmetric_init(model, mode=cfg.init_mode)
            mlflow.log_params({"init_mode": cfg.init_mode})

            # init weights 저장 → eval 시 base weight 보정 + 삼각 측량에 사용
            if init_info["init_weights"]:
                init_path = Path(cfg.output_dir) / "init_weights.safetensors"
                _save_safetensors(init_info["init_weights"], str(init_path))
                print(f"[sym_lora] init weights 저장: {init_path}")

            # sanity check: W_before - W_after ≈ expected_delta
            _sanity_check_init(model, init_info, w_before)

        # Residual B: LoRA 모듈 간 residual connection
        if cfg.residual_b:
            from src.models.residual_lora import apply_residual_lora
            apply_residual_lora(model)
            mlflow.log_param("residual_b", True)

        # A freeze: gradient 차단
        if cfg.freeze_a:
            n_frozen = 0
            for name, module in model.named_modules():
                if not hasattr(module, "lora_A") or len(module.lora_A) == 0:
                    continue
                for key in module.lora_A:
                    module.lora_A[key].weight.requires_grad_(False)
                    n_frozen += 1
            print(f"[freeze_a] {n_frozen} lora_A modules frozen")
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"[freeze_a] trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
            mlflow.log_param("freeze_a", True)

        # init 직후 ‖A‖, ‖B‖ 통계 로깅 (standard 포함 모든 모드)
        norm_stats = _log_lora_norms(model, tag="init")
        for k, v in norm_stats.items():
            mlflow.log_metric(k.replace("/", "_"), v)

        print("데이터셋 로드 중...")
        dataset = load_from_disk(cfg.data_path)
        train_ds, eval_ds = dataset["train"], dataset["test"]
        mlflow.log_metric("train_samples", len(train_ds))
        mlflow.log_metric("eval_samples", len(eval_ds))

        print(f"학습 시작: train={len(train_ds)}, eval={len(eval_ds)}")
        trainer = build_trainer(model, tokenizer, train_ds, eval_ds, cfg)
        trainer.train()

        print(f"어댑터 저장: {cfg.output_dir}")
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        # Residual R matrices 저장 (PEFT save_pretrained는 LoRA만 저장)
        if cfg.residual_b:
            from safetensors.torch import save_file as _save_safetensors
            r_weights = {}
            for name, module in model.named_modules():
                if hasattr(module, "residual_R"):
                    r_weights[f"{name}.residual_R.weight"] = module.residual_R.weight.detach().cpu().contiguous()
            r_path = Path(cfg.output_dir) / "residual_R.safetensors"
            _save_safetensors(r_weights, str(r_path))
            print(f"[residual_b] R weights 저장: {r_path} ({len(r_weights)} modules)")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            mlflow.log_metric("peak_vram_gb", round(vram_gb, 2))
            print(f"피크 VRAM: {vram_gb:.2f} GB")

    print("학습 완료")
