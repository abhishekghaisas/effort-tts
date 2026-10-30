import json
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.utils import setup_logger

logger = setup_logger(__name__)


class ChatterboxDataset(Dataset):
    """Stock dataset (unchanged): conditioning comes from the same clip as the target."""

    def __init__(self, config):
        self.cfg = config
        self.preprocessed_dir = config.preprocessed_dir

        if not os.path.exists(self.preprocessed_dir):
            raise FileNotFoundError(f"Preprocessing folder not found: {self.preprocessed_dir}.")

        self.files = [f for f in os.listdir(self.preprocessed_dir) if f.endswith(".pt")]

        if len(self.files) == 0:
            raise RuntimeError(f"There are no .pt files in the folder: {self.preprocessed_dir}")

        logger.info(f"Dataset loaded. Total sample: {len(self.files)}")

        self.sot_token = config.start_text_token
        self.eot_token = config.stop_text_token

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            filename = self.files[idx]

            pt_path = os.path.join(self.preprocessed_dir, filename)

            data = torch.load(pt_path)

            text_tokens = data["text_tokens"]
            if text_tokens.size(0) > self.cfg.max_text_len - 2:
                text_tokens = text_tokens[: self.cfg.max_text_len - 2]

            sot = torch.tensor([self.sot_token], dtype=torch.long)
            eot = torch.tensor([self.eot_token], dtype=torch.long)
            text_tokens = torch.cat([sot, text_tokens, eot])

            speech_tokens = data["speech_tokens"]
            if speech_tokens.size(0) > self.cfg.max_speech_len:
                speech_tokens = speech_tokens[: self.cfg.max_speech_len]

            speaker_emb = data["speaker_emb"]
            prompt_tokens = data["prompt_tokens"]

            if random.random() < 0.20:
                speaker_emb = torch.zeros_like(speaker_emb)
                prompt_tokens = torch.zeros(1, dtype=torch.long)

            return {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
            }

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None


class EffortDataset(Dataset):
    """Effort-conditioned dataset.

    Each manifest line gives a target clip id, its effort label and a pool of reference clip ids of the SAME
    speaker. Text and speech tokens come from the target; speaker_emb and prompt_tokens come from a randomly
    chosen reference clip (never the target itself), so the only reliable source of effort is the scalar.
    """

    def __init__(self, config):
        self.cfg = config
        self.preprocessed_dir = config.preprocessed_dir

        if not os.path.exists(self.preprocessed_dir):
            raise FileNotFoundError(f"Preprocessing folder not found: {self.preprocessed_dir}.")
        if not os.path.exists(config.effort_manifest):
            raise FileNotFoundError(f"Effort manifest not found: {config.effort_manifest}")

        have = {f[:-3] for f in os.listdir(self.preprocessed_dir) if f.endswith(".pt")}
        self.items = []
        skipped = 0
        with open(config.effort_manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                refs = [r for r in rec.get("refs", []) if r != rec["id"] and r in have]
                if rec["id"] not in have or not refs:
                    skipped += 1
                    continue
                self.items.append({"id": rec["id"], "refs": refs, "effort": float(rec["effort"])})

        if len(self.items) == 0:
            raise RuntimeError(
                "No usable items: check that preprocessing finished and ids match the manifest."
            )

        efforts = torch.tensor([x["effort"] for x in self.items])
        logger.info(
            f"EffortDataset: {len(self.items)} targets ({skipped} skipped); effort label "
            f"min {efforts.min():.2f}, median {efforts.median():.2f}, max {efforts.max():.2f}"
        )

        self.sot_token = config.start_text_token
        self.eot_token = config.stop_text_token

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            tgt = torch.load(os.path.join(self.preprocessed_dir, f"{item['id']}.pt"))
            ref = torch.load(
                os.path.join(self.preprocessed_dir, f"{random.choice(item['refs'])}.pt")
            )

            text_tokens = tgt["text_tokens"]
            if text_tokens.size(0) > self.cfg.max_text_len - 2:
                text_tokens = text_tokens[: self.cfg.max_text_len - 2]
            sot = torch.tensor([self.sot_token], dtype=torch.long)
            eot = torch.tensor([self.eot_token], dtype=torch.long)
            text_tokens = torch.cat([sot, text_tokens, eot])

            speech_tokens = tgt["speech_tokens"]
            if speech_tokens.size(0) > self.cfg.max_speech_len:
                speech_tokens = speech_tokens[: self.cfg.max_speech_len]

            speaker_emb = ref["speaker_emb"]
            prompt_tokens = ref["prompt_tokens"]

            if random.random() < self.cfg.cond_dropout:
                speaker_emb = torch.zeros_like(speaker_emb)
                prompt_tokens = torch.zeros(1, dtype=torch.long)

            return {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
                "effort": torch.tensor(item["effort"], dtype=torch.float32),
            }

        except Exception as e:
            logger.error(f"Error loading {item['id']}: {e}")
            return None


def data_collator_standart(batch):

    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    # Padding
    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_tokens = pad_sequence(
        [x["speech_tokens"] for x in batch], batch_first=True, padding_value=0
    )
    prompt_tokens = pad_sequence(
        [x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0
    )

    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    # Lengths
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens,
    }


def data_collator_effort(batch):
    """Standard collator plus the per-clip effort scalar."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    out = data_collator_standart(batch)
    out["effort"] = torch.stack([x["effort"] for x in batch])
    return out


def data_collator_turbo(batch):

    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    # 1. Text Tokens Padding
    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)

    # 2. Speech Tokens Padding
    speech_tokens = pad_sequence(
        [x["speech_tokens"] for x in batch], batch_first=True, padding_value=0
    )
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

    # 3. Prompt Tokens Padding
    prompt_tokens = pad_sequence(
        [x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0
    )
    prompt_lens = torch.tensor([x["prompt_tokens"].shape[0] for x in batch], dtype=torch.long)

    # 4. Speaker Embedding
    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens,
        "prompt_lens": prompt_lens,
    }
