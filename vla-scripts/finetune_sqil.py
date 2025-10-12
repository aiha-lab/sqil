"""
finetune_sqil.py
LoRA fine-tuning with precomputed saliency weighting 
"""
import draccus
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import tqdm
import torch
import torch.nn.functional as F
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from einops import rearrange
import wandb
import numpy as np

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PerturbationAttention:
    """Perturbation-based saliency visualization (per-sample scores). Used ONLY for precompute."""
    def __init__(self, model, image_size=[224, 224], patch_size=[28, 28], device="cpu"):
        self.model = model
        self.device = device
        H, W = image_size
        h, w = patch_size
        nh, nw = H // h, W // w
        self.num_patches = nh * nw
        eye = torch.eye(self.num_patches, device=device).view(self.num_patches, self.num_patches, 1, 1)
        mask = eye.repeat(1, 1, h, w)
        mask = rearrange(mask.view(self.num_patches, nh, nw, h, w), "p nh nw h w -> p (nh h) (nw w)")
        self.mask = mask
        self.H, self.W = H, W

    @torch.inference_mode()
    def __call__(self, batch) -> List[float]:
        """Return per-sample saliency score list (len == batch size)."""
        device = self.device
        rgb = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
        B, C, H, W = rgb.shape
        assert (H, W) == (self.H, self.W)

        vstart = self.model.vision_backbone.featurizer.patch_embed.num_patches
        rgb_mean = rgb.mean(dim=(2, 3), keepdim=True)
        running = torch.zeros(B, device=device, dtype=torch.bfloat16)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            base_out: CausalLMOutputWithPast = self.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=rgb,
            )
            base_logits = base_out.logits[:, vstart:-1]

            for j in range(self.num_patches):
                m = self.mask[j].to(device=device, dtype=torch.bfloat16).view(1, 1, H, W)
                pv = rgb_mean * m + rgb * (1.0 - m)
                out_j: CausalLMOutputWithPast = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=pv,
                )
                logits_j = out_j.logits[:, vstart:-1]
                diff2 = (logits_j - base_logits).pow(2).mean((1, 2))
                running += diff2

        scores = (running / float(self.num_patches)).float().detach().cpu().tolist()
        return [float(s) for s in scores]


@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    dataset_name: str = "droid_wipe"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 2e-5
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000
    save_latest_checkpoint_only: bool = True
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    wandb_project: str = "openvla"
    wandb_entity: str = "parks"
    run_id_note: Optional[str] = None
    precompute: bool = False
    saliency_cache_path: Optional[Path] = None
    debug_one_sample: bool = False  # quick single-sample smoke test
    sis_percentile: float = 80.0    # percentile for thresholding (top 20% => 80.0)


# -------------------- UID-aware cache helpers --------------------

def _save_uid_scores_jsonl(path: Path, uids: List[str], scores: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for uid, s in zip(uids, scores):
            f.write(json.dumps({"uid": uid, "score": float(s)}) + "\n")


def _load_sis_map_and_threshold(path: Path, percentile: float) -> Tuple[Dict[str, float], float]:
    sis_map: Dict[str, float] = {}
    all_scores: List[float] = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            uid = obj["uid"]
            s = float(obj["score"])
            sis_map[uid] = s
            all_scores.append(s)
    if len(all_scores) == 0:
        raise RuntimeError("Saliency cache is empty.")
    threshold_val = float(np.percentile(np.asarray(all_scores, dtype=np.float32), percentile))
    return sis_map, threshold_val


@torch.no_grad()
def _sis_weights_per_sample(
    uids: List[str],
    sis_map: Dict[str, float],
    threshold: float,
    default: Optional[float] = None,
) -> torch.Tensor:
    """
    Return per-sample tensor of SIS weights (2.0 or 1.0) matching batch order.
    """
    ws: List[float] = []
    for uid in uids:
        s = sis_map.get(uid, default)
        w = 2.0 if (s is not None and s >= threshold) else 1.0
        ws.append(w)
    return torch.tensor(ws, dtype=torch.float32, device="cuda")  # [B]


# -------------------- Precompute phase (offline) --------------------

def precompute_saliency(cfg: FinetuneConfig):
    device = "cuda"
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b-finetuned-libero-10",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=False,
    )
    # IMPORTANT: collator must preserve "uid" in batch
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    bs = 1 if cfg.debug_one_sample else cfg.batch_size
    loader = DataLoader(dataset, batch_size=bs, collate_fn=collator, num_workers=0)

    viz = PerturbationAttention(model, image_size=list(model.config.image_sizes), patch_size=[28, 28], device=device)

    out_path = cfg.saliency_cache_path or (cfg.run_root_dir / f"{cfg.dataset_name}_saliency.jsonl")
    if out_path.exists():
        out_path.unlink()  # fresh cache

    for batch in tqdm.tqdm(loader):
        uids: List[str] = batch.get("uid", None)
        if uids is None:
            raise RuntimeError("Batch has no 'uid'. Ensure collator preserves 'uid' in the batch.")
        scores = viz(batch)  # per-sample scores
        _save_uid_scores_jsonl(out_path, uids, scores)
        if cfg.debug_one_sample:
            break

    print(f"[precompute] Saved UID->score JSONL to {out_path}")


# -------------------- Finetune phase  --------------------

@draccus.wrap()
def finetune(cfg: FinetuneConfig):
    if cfg.precompute:
        precompute_saliency(cfg)
        return

    print(f"Fine-tuning `{cfg.vla_path}` on `{cfg.dataset_name}` (offline SIS)")
    assert torch.cuda.is_available()
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla_fp = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b-finetuned-libero-10",   #libero-spatial, libero-object, libero-goal, libero-10
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla, vla_fp = vla.to(device_id), vla_fp.to(device_id)
    for p in vla_fp.parameters():
        p.requires_grad = False

    if cfg.use_lora:
        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_cfg)

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    optimizer = AdamW([p for p in vla.parameters() if p.requires_grad], lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    save_dataset_statistics(dataset.dataset_statistics, cfg.run_root_dir)

    # IMPORTANT: collator must preserve "uid" in batch
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_bs = 1 if cfg.debug_one_sample else cfg.batch_size
    dataloader = DataLoader(dataset, batch_size=train_bs, collate_fn=collator, num_workers=0)

    # Load SIS cache and compute threshold (top 20% by default)
    if not cfg.saliency_cache_path or not os.path.exists(cfg.saliency_cache_path):
        raise FileNotFoundError("saliency_cache_path is missing for offline SIS. Run with --precompute first.")
    sis_map, threshold_val = _load_sis_map_and_threshold(cfg.saliency_cache_path, cfg.sis_percentile)
    # print(f"[finetune] Loaded {len(sis_map)} UID scores; threshold(p{cfg.sis_percentile:.1f})={threshold_val:.6f}")

    recent_losses, recent_accs, recent_l1, recent_qrd = (
        deque(maxlen=cfg.grad_accumulation_steps),
        deque(maxlen=cfg.grad_accumulation_steps),
        deque(maxlen=cfg.grad_accumulation_steps),
        deque(maxlen=cfg.grad_accumulation_steps),
    )
    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{cfg.dataset_name}")

    with tqdm.tqdm(total=(1 if cfg.debug_one_sample else cfg.max_steps), leave=False) as progress:
        vla.train()
        vla_fp.eval()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                output_fp = vla_fp(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                )

                out_loss = output.loss
                qrd_diff = (output.logits - output_fp.logits).pow(2).mean(dim=-1)  # [B, T]

                # Per-sample SIS weights from cache (2 if >= threshold else 1)
                uids: List[str] = batch.get("uid", None)
                if uids is None:
                    raise RuntimeError("Batch has no 'uid'. Ensure collator preserves 'uid' in the batch.")
                sis_w = _sis_weights_per_sample(uids, sis_map, threshold_val)  # [B]

                qrd_loss = (sis_w[:, None] * qrd_diff).mean()  # weighted mean over batch × time
                loss = out_loss + qrd_loss

            (loss / cfg.grad_accumulation_steps).backward()

            # Metrics
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            acc = ((action_preds == action_gt) & mask).sum().float() / mask.sum().float()

            actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
            actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
            l1 = F.l1_loss(actions_pred, actions_gt)

            recent_losses.append(out_loss.item())
            recent_accs.append(acc.item())
            recent_l1.append(l1.item())
            recent_qrd.append(qrd_loss.item())

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()
                if cfg.debug_one_sample:
                    break

            step = batch_idx // cfg.grad_accumulation_steps
            if step % 10 == 0:
                wandb.log(
                    {
                        "train_loss": sum(recent_losses) / len(recent_losses),
                        "action_accuracy": sum(recent_accs) / len(recent_accs),
                        "l1_loss": sum(recent_l1) / len(recent_l1),
                        "qrd_loss": sum(recent_qrd) / len(recent_qrd),
                        "sis_w_mean": sis_w.mean().item(),
                    },
                    step=step,
                )

            if step >= cfg.max_steps:
                print("Max steps reached.")
                break


if __name__ == "__main__":
    finetune()

