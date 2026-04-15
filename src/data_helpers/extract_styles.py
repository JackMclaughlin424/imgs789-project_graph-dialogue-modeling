"""
Extract WavLM style embeddings (GST proxy) and word-level BERT embeddings
from a merged HDF5 audio dataset, writing outputs to a companion features HDF5.

Outputs per audio row (hdf5_idx >= 0):
  gst/{idx:06d}       float32 [768]         WavLM layer-N mean-pooled embedding
  bert/{idx:06d}      float32 [n_words, 768] word-level BERT embeddings

Outputs per text-only row (hdf5_idx == -1, keyed by parquet integer row index):
  bert_text/{row_idx} float32 [n_words, 768]

Usage:
  python extract_styles.py path/to/merged_audio.h5
  python extract_styles.py path/to/merged_audio.h5 --layer 9 --device cuda:1
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    WavLMModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModel,
)


def load_models(device: torch.device, wavlm_layer: int):
    print(f"Loading WavLM on {device}...")
    wavlm_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device).eval()

    print(f"Loading BERT on {device}...")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    return wavlm_extractor, wavlm_model, bert_tokenizer, bert_model


@torch.no_grad()
def extract_wavlm_gst(
    waveform: np.ndarray,
    extractor,
    model,
    device: torch.device,
    layer: int,
) -> np.ndarray:
    inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer].mean(dim=1).squeeze(0).cpu().numpy().astype("float32")


@torch.no_grad()
def extract_bert_words(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray | None:
    clean = "".join(c for c in text.lower() if c in "abcdefghijklmnopqrstuvwxyz' ")
    words = [w for w in clean.split() if w]
    if not words:
        return None

    inputs = tokenizer(clean, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # strip [CLS] and [SEP] tokens from the sequence
    hidden = outputs.last_hidden_state[0][1:-1]

    result = []
    cursor = 0
    for word in words:
        subwords = tokenizer.tokenize(word)
        if not subwords:
            continue
        n = len(subwords)
        result.append(hidden[cursor : cursor + n].mean(dim=0).cpu().numpy())
        cursor += n

    return np.stack(result).astype("float32") if result else None


def derive_parquet_path(h5_path: Path) -> Path:
    stem = h5_path.stem.replace("audio", "metadata")
    return h5_path.parent / f"{stem}.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=Path, help="Path to merged_audio*.h5")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Metadata parquet path (default: derived from h5_path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output features HDF5 path (default: <dir>/<stem replacing 'audio' with 'features'>.h5)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="WavLM hidden layer index to use as GST proxy (default: 6)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda', 'cuda:1', 'cpu' (default: cuda if available)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device      : {device}")

    parquet_path = args.parquet or derive_parquet_path(args.h5_path)
    output_path = args.output or (
        args.h5_path.parent / f"{args.h5_path.stem.replace('audio', 'features')}.h5"
    )

    print(f"Audio HDF5  : {args.h5_path}")
    print(f"Metadata    : {parquet_path}")
    print(f"Output      : {output_path}")
    print(f"WavLM layer : {args.layer}")

    df = pd.read_parquet(parquet_path)
    audio_rows = df[df["hdf5_idx"] >= 0]
    text_rows = df[df["hdf5_idx"] == -1]
    print(f"Audio rows  : {len(audio_rows):,}")
    print(f"Text-only   : {len(text_rows):,}")

    wavlm_extractor, wavlm_model, bert_tokenizer, bert_model = load_models(device, args.layer)

    with h5py.File(args.h5_path, "r") as src, h5py.File(output_path, "a") as dst:
        for grp in ("gst", "bert", "bert_text"):
            if grp not in dst:
                dst.create_group(grp)

        dst.attrs["wavlm_layer"] = args.layer
        dst.attrs["wavlm_model"] = "microsoft/wavlm-base"
        dst.attrs["bert_model"] = "bert-base-uncased"

        for _, row in tqdm(audio_rows.iterrows(), total=len(audio_rows), desc="Audio rows"):
            idx = int(row["hdf5_idx"])
            key = f"{idx:06d}"
            need_gst = key not in dst["gst"]
            need_bert = key not in dst["bert"]
            if not need_gst and not need_bert:
                continue

            waveform = src["audio"][key][()]

            if need_gst:
                gst = extract_wavlm_gst(waveform, wavlm_extractor, wavlm_model, device, args.layer)
                dst["gst"].create_dataset(key, data=gst)

            if need_bert and pd.notna(row.get("transcription")):
                bert = extract_bert_words(str(row["transcription"]), bert_tokenizer, bert_model, device)
                if bert is not None:
                    dst["bert"].create_dataset(key, data=bert)

        for row_idx, row in tqdm(text_rows.iterrows(), total=len(text_rows), desc="Text-only rows"):
            key = str(row_idx)
            if key in dst["bert_text"] or pd.isna(row.get("transcription")):
                continue
            bert = extract_bert_words(str(row["transcription"]), bert_tokenizer, bert_model, device)
            if bert is not None:
                dst["bert_text"].create_dataset(key, data=bert)

    print(f"Done. Features written to {output_path}")


if __name__ == "__main__":
    main()
