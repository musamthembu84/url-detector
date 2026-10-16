"""
Phase 3: Parameter-Efficient Model Architecture.

Wraps loading the foundation classification model and shows where a LoRA
adapter would be attached for cheap fine-tuning. This module does not run a
full training loop (that needs labeled gold-standard data); it demonstrates
the architecture described in the lifecycle doc: freeze the base model,
inject low-rank adapters into the attention blocks, and keep the existing
4-way classification head.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.logging_config import get_logger

logger = get_logger(__name__)

MODEL_NAME = "Eason918/malicious-url-detector-v2"


@dataclass(frozen=True)
class LoRAConfig:
    """Minimal LoRA hyperparameters for attention-block adapters."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "key", "value")


def load_foundation_model(model_name: str = MODEL_NAME):
    """Load the pretrained tokenizer + classification head, base params frozen.

    Freezing here mirrors the lifecycle doc's PEFT step: the 6-layer
    distilroberta backbone keeps its pretrained weights, and only a LoRA
    adapter (attached separately via apply_lora_adapter) would be trainable.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    logger.info("Loaded foundation model '%s'", model_name)

    frozen = 0
    for param in model.base_model.parameters():
        param.requires_grad = False
        frozen += param.numel()
    logger.info("Froze %d base-model params (classification head stays trainable)", frozen)

    return tokenizer, model


def apply_lora_adapter(model, config: LoRAConfig = LoRAConfig()):
    """Attach LoRA adapters to the model's attention blocks, if peft is available.

    Requires the `peft` package (pip install peft). Kept optional so the
    rest of the pipeline (ingestion/inference) runs without pulling in an
    extra training-only dependency.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        logger.error("peft not installed; cannot attach LoRA adapter")
        raise RuntimeError(
            "peft is not installed. Run `pip install peft` to enable "
            "LoRA fine-tuning (only needed for retraining, not inference)."
        ) from exc

    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        # PEFT freezes every non-LoRA param by default, which would also
        # freeze the 4-way classification head. modules_to_save keeps it
        # trainable alongside the LoRA matrices, since the head itself is
        # part of what we need fine-tuning to adjust.
        modules_to_save=["classifier"],
        bias="none",
        task_type="SEQ_CLS",
    )
    logger.info(
        "Attaching LoRA adapter: r=%d alpha=%d dropout=%.2f targets=%s",
        config.r, config.alpha, config.dropout, config.target_modules,
    )
    return get_peft_model(model, lora_config)


if __name__ == "__main__":
    from pipeline.logging_config import setup_logging

    setup_logging()
    tok, mdl = load_foundation_model()
    trainable = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    total = sum(p.numel() for p in mdl.parameters())
    print(f"Loaded {MODEL_NAME}: {trainable}/{total} params trainable before LoRA")
