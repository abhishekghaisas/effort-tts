import os
import sys

import torch
from safetensors.torch import save_file
from src.chatterbox_.models.t3.t3 import T3
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.config import TrainConfig
from src.dataset import (
    ChatterboxDataset,
    EffortDataset,
    data_collator_effort,
    data_collator_standart,
    data_collator_turbo,
)
from src.inference_callback import InferenceCallback
from src.model import ChatterboxTrainerWrapper, resize_and_load_t3_weights
from src.preprocess_file_based import preprocess_dataset_file_based
from src.preprocess_json import preprocess_dataset_json_based
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.utils import check_pretrained_models, setup_logger
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxFinetune")


def main():

    cfg = TrainConfig()

    logger.info("--- Starting Chatterbox Finetuning ---")
    logger.info(f"Mode: {'CHATTERBOX-TURBO' if cfg.is_turbo else 'CHATTERBOX-TTS'}")
    logger.info(f"Training Strategy: {'LoRA' if cfg.is_lora else 'Full Fine-Tune'}")

    use_effort = bool(getattr(cfg, "effort_manifest", None))
    if use_effort and cfg.is_turbo:
        logger.error(
            "Effort conditioning needs the standard model (Turbo has no emotion_adv). Set is_turbo = False."
        )
        sys.exit(1)
    logger.info(f"Effort conditioning: {use_effort}")

    # 0. CHECK MODEL FILES
    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. SELECT THE CORRECT ENGINE CLASS
    if cfg.is_turbo:
        EngineClass = ChatterboxTurboTTS
    else:
        EngineClass = ChatterboxTTS

    logger.info(f"Device: {device}")
    logger.info(f"Model Directory: {cfg.model_dir}")

    # 2. LOAD ORIGINAL MODEL TEMPORARILY
    logger.info("Loading original model to extract weights...")
    # Loading on CPU first to save VRAM
    tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")

    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    # 3. CREATE NEW T3 MODEL WITH NEW VOCAB SIZE
    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")

    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size

    # Prevent caching during training
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False
    else:
        new_t3_config.use_cache = False

    new_t3_model = T3(hp=new_t3_config)

    # 4. TRANSFER WEIGHTS
    logger.info("Transferring weights...")
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)

    if cfg.is_turbo:
        logger.info("Turbo Mode: Removing backbone WTE layer...")
        if hasattr(new_t3_model.tfmr, "wte"):
            del new_t3_model.tfmr.wte

    # Clean up memory
    del tts_engine_original
    del pretrained_t3_state_dict

    # 5. PREPARE ENGINE FOR TRAINING
    tts_engine_new = EngineClass.from_local(cfg.model_dir, device="cpu")
    tts_engine_new.t3 = new_t3_model

    logger.info("Freezing S3Gen and VoiceEncoder...")
    for param in tts_engine_new.ve.parameters():
        param.requires_grad = False

    for param in tts_engine_new.s3gen.parameters():
        param.requires_grad = False

    # 6. APPLY LORA OR FULL FINE-TUNE
    if cfg.is_lora:
        # --- LoRA PATH ---
        logger.info("Applying LoRA configuration...")

        for param in tts_engine_new.t3.parameters():
            param.requires_grad = False

        from peft import LoraConfig, get_peft_model

        logger.info(f"LoRA Target Modules: {cfg.lora_target_modules}")
        logger.info(f"Modules to Full Train (Embeddings): {cfg.lora_modules_to_save}")

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.turbo_lora_target_modules
            if cfg.is_turbo
            else cfg.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=cfg.lora_modules_to_save,
        )

        tts_engine_new.t3 = get_peft_model(tts_engine_new.t3, peft_config)
        tts_engine_new.t3.print_trainable_parameters()

    else:
        # --- FULL FINE-TUNE PATH ---
        logger.info("Full fine-tune: enabling all T3 parameters...")
        tts_engine_new.t3.train()
        for param in tts_engine_new.t3.parameters():
            param.requires_grad = True

    # 7. PREPROCESSING
    if cfg.preprocess:
        logger.info("Initializing Preprocess dataset...")

        if cfg.ljspeech:
            preprocess_dataset_ljspeech(cfg, tts_engine_new)
        elif cfg.json_format:
            preprocess_dataset_json_based(cfg, tts_engine_new)
        else:
            preprocess_dataset_file_based(cfg, tts_engine_new)
    else:
        logger.info("Skipping the preprocessing dataset step...")

    # 8. DATASET & WRAPPER
    logger.info("Initializing Dataset...")
    train_ds = EffortDataset(cfg) if use_effort else ChatterboxDataset(cfg)

    trainer_callbacks = []
    if cfg.is_inference:
        inference_cb = InferenceCallback(cfg)
        trainer_callbacks.append(inference_cb)

    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)
    model_wrapper.ref_conditioning = bool(use_effort and getattr(cfg, "ref_conditioning", False))
    logger.info(f"Reference conditioning (no prompt masking): {model_wrapper.ref_conditioning}")

    if cfg.is_turbo:
        logger.info("Using Turbo Data Collator (with dynamic prompt masking)")
        selected_collator = data_collator_turbo
    elif use_effort:
        logger.info("Using Effort Data Collator")
        selected_collator = data_collator_effort
    else:
        logger.info("Using Standard Data Collator")
        selected_collator = data_collator_standart

    # 9. TRAINING ARGUMENTS
    # bf16 needs compute capability >= 8 (L4, A100); a T4 (7.5) must use fp16 with loss scaling.
    cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    use_bf16 = cap >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16
    logger.info(
        f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32 (no GPU)'} (GPU capability {cap})"
    )
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,  # Required for our custom wrapper
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=use_fp16,
        bf16=use_bf16,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=True,  # Reduces VRAM usage by ~60%
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=selected_collator,
        callbacks=trainer_callbacks,
    )

    logger.info("Starting Training Loop...")
    last_ckpt = get_last_checkpoint(cfg.output_dir) if os.path.isdir(cfg.output_dir) else None
    if last_ckpt:
        logger.info(f"Resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # 10. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.is_lora:
        # Save LoRA adapter + resized embeddings via PEFT
        save_path = os.path.join(cfg.output_dir, "new_lang_adapter")
        tts_engine_new.t3.save_pretrained(save_path)
        logger.info(f"LoRA adapter saved to: {save_path}")
        logger.info("NOTE: This adapter contains both LoRA weights AND the new resized embeddings.")
    else:
        # Save full model weights as safetensors
        filename = "t3_turbo_finetuned.safetensors" if cfg.is_turbo else "t3_finetuned.safetensors"
        final_model_path = os.path.join(cfg.output_dir, filename)
        save_file(tts_engine_new.t3.state_dict(), final_model_path)
        logger.info(f"Full model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
