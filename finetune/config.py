from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Effort-conditioned pilot config (replaces src/config.py of chatterbox-finetuning).

    Changes from the stock file are marked with  # EFFORT.
    """

    # --- Paths ---
    model_dir: str = "./pretrained_models"

    # Unused in file-based mode, kept for compatibility with the stock scripts.
    csv_path: str = "./MyTTSDataset/metadata.csv"

    # EFFORT: folder of ID.wav + ID.txt pairs (targets AND reference clips), made by make_effort_dataset.py
    wav_dir: str = "./EffortDataset/wavs"
    preprocessed_dir = "./EffortDataset/preprocess"

    # EFFORT: one JSON object per line: {"id": ..., "effort": 0.5-2.0, "refs": [ids of same-speaker reference clips]}
    effort_manifest: str = "./EffortDataset/effort_manifest.jsonl"

    output_dir: str = "./chatterbox_output"

    is_inference = False
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "Watch out! Everybody get back from the edge right now!"

    ljspeech = False  # EFFORT: file-based format
    json_format = False
    preprocess = True  # set False after the first successful preprocessing run

    is_turbo: bool = False  # EFFORT: must stay False (Turbo has no emotion_adv slot)
    is_lora: bool = True

    lora_r: int = 32  # EFFORT: smaller than stock (128) to limit drift in a pilot
    lora_alpha: int = 64
    turbo_lora_target_modules: list[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "c_fc", "spkr_enc"]
    )
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "spkr_enc",
        ]
    )
    # EFFORT: train the scalar projection fully; keep text embeddings frozen (English vocab is unchanged).
    # If merge_lora.py complains, add "text_emb", "text_head" back.
    lora_modules_to_save: list[str] = field(default_factory=lambda: ["emotion_adv_fc"])

    # --- Vocabulary ---
    # EFFORT: 704 = the original English tokenizer.json (NOT the multilingual one that setup.py saves as
    # tokenizer.json). Keeping 704 means no embedding resize, English token ids stay valid, and the merged
    # T3 can be dropped into stock chatterbox-tts for evaluation.
    new_vocab_size: int = 52260 if is_turbo else 704

    # --- Effort conditioning (EFFORT) ---
    ref_conditioning = True  # speaker_emb + prompt come from a DIFFERENT clip of the same speaker
    cond_dropout = 0.20  # chance of zeroing speaker_emb and prompt (as in the stock dataset)

    # --- Hyperparameters ---
    batch_size: int = (
        4  # EFFORT: T4-friendly; effective batch = 4 * 4 = 16 (if out of memory: 2 and 8)
    )
    grad_accum: int = 4
    learning_rate: float = 1e-4 if is_lora else 1e-5
    num_epochs: int = 12 if is_lora else 30

    save_steps: int = 100
    save_total_limit: int = 2  # each checkpoint holds the full T3 (~2 GB)
    dataloader_num_workers: int = 2  # Colab VMs have about 2 CPUs

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850
    prompt_duration: float = (
        6.0  # EFFORT: 6 s = 150 tokens, matching inference (stock training used 3 s)
    )
