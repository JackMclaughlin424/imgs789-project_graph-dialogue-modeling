"""
build_hdf5.py

Usage - Default data should be in place if you've run the prerequisite files
-----
  python build_hdf5.py 
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import math
from scipy.signal import resample_poly

# Columns with long free-text – kept in Parquet, not HDF5 attrs
TEXT_COLUMNS = {
    "text_description", "transcription",
    "intrinsic_tags", "situational_tags",
    "basic_tags", "all_tags",
}

# Columns to store as lightweight HDF5 dataset-level attrs (short scalars)
ATTR_COLUMNS = {
    "source", "relative_audio_path", "speakerid", "name", "gender",
    "accent", "pitch", "speaking_rate", "noise",
    "utterance_pitch_mean", "snr", "tag_of_interest",
    "conv_id", "turn_index", "prev_filename", "record_type",
}

# hdf5_idx sentinel values written to metadata.parquet
IDX_TEXT_ONLY = -1   # record_type == 'text_only': no audio, intentional
IDX_ERROR     = -2   # audio expected but missing or unreadable



def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (waveform float32, sample_rate). Mixes down to mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)          # (samples,)
    return mono, sr


def safe_attr(value):
    """Convert a value to something HDF5 attrs will accept."""
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)

def resample_waveform(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    # resample_poly needs integer up/down factors, so reduce the ratio first
    gcd = math.gcd(orig_sr, target_sr)
    up   = target_sr // gcd
    down = orig_sr   // gcd
    resampled = resample_poly(waveform, up, down)
    return resampled.astype(np.float32)


def build(df_path: str, audio_root_PSC: str, audio_root_ST: str,
          out_h5: str, out_meta: str, resample_rate: int | None
          # Debugging/test parameters
          , DEBUG_MAX_ROW, DEBUG_MAX_TURNS, DEBUG_PERCENT_EXPRESSO, SEED):

    # resolve() fixes max path length errors
    df_path        = Path(df_path).resolve()
    audio_root_PSC = Path(audio_root_PSC).resolve()
    audio_root_ST  = Path(audio_root_ST).resolve()


    if DEBUG_MAX_ROW % DEBUG_MAX_TURNS > 0 and DEBUG_MAX_ROW >= 0 and DEBUG_MAX_TURNS >= 0:
        # round down so we don't cut a conversation mid-turn
        turns_per_convo = DEBUG_MAX_TURNS  
        DEBUG_MAX_ROW = (DEBUG_MAX_ROW // turns_per_convo) * turns_per_convo
        print(f"  DEBUG_MAX_ROW adjusted to {DEBUG_MAX_ROW:,} (nearest multiple of {turns_per_convo} turns)")


    h5_path = Path(out_h5)
    out_h5         = h5_path  if DEBUG_MAX_ROW < 0 else h5_path.parent / Path(h5_path.stem + f"_{str(DEBUG_MAX_ROW)}" + h5_path.suffix)
    meta_path = Path(out_meta)
    out_meta       = meta_path if DEBUG_MAX_ROW < 0 else meta_path.parent / Path(meta_path.stem + f"_{str(DEBUG_MAX_ROW)}" + meta_path.suffix)

    SOURCE_ROOTS = {
        "expresso": audio_root_PSC,
        "styletalk":  audio_root_ST,
    }

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    if resample_rate is not None:
        print(f"Resampling enabled: target {resample_rate} Hz  (files with native SR < target will error)")
 

    # Load DataFrame
    print(f"Loading DataFrame from {df_path} ...")
    df = pd.read_parquet(df_path) if df_path.suffix == ".parquet" else pd.read_csv(df_path)
    n_text_only_expected = (df["record_type"] == "text_only").sum()
    print(f"  {len(df):,} rows total  |  "
          f"text_only: {n_text_only_expected:,}  |  "
          f"audio rows: {len(df) - n_text_only_expected:,}")

    attr_cols_present = ATTR_COLUMNS & set(df.columns)

    # used in testing: Filtering max turns 
    max_turn_idx = DEBUG_MAX_TURNS - 1

    if DEBUG_MAX_TURNS >= 0 and "turn_index" in df.columns:
        # keeps only utterances from short-enough conversations, -1 means no filter
        df = df[df["turn_index"] <= max_turn_idx].copy() # turn_index is 0-based
        print(f"  After DEBUG_MAX_TURNS={DEBUG_MAX_TURNS} filter: {len(df):,} rows remain")


    
    # filter to only full conversations
    
    if DEBUG_MAX_ROW >= 0 and DEBUG_MAX_TURNS >= 0:
        n_convos = DEBUG_MAX_ROW // (DEBUG_MAX_TURNS)
        n_expresso  = round(n_convos * DEBUG_PERCENT_EXPRESSO)
        n_styletalk = n_convos - n_expresso  # remainder avoids floating point drift
        
        collected = set()
        for source_name, n in [("expresso", n_expresso), ("styletalk", n_styletalk)]:

            # anchor on rows that are the max turn given my filters, then walk backwards to build conversations
            last_turns = (
                df[(df["turn_index"] == max_turn_idx) & (df["source"] == source_name)]
                .sample(n=n, random_state=SEED)  # random so we don't always get the same convos
            )
            for _, row in last_turns.iterrows():
                current = row
                while current is not None:
                    collected.add(current["relative_audio_path"])
                    prev = current["prev_filename"]
                    if pd.isna(prev) or prev == "":
                        break
                    match = df[df["relative_audio_path"] == prev]
                    current = match.iloc[0] if not match.empty else None

        df = df[df["relative_audio_path"].isin(collected)].copy()
        print(f"  Filtered to {n_convos} conversations "
            f"({n_expresso} expresso, {n_styletalk} styletalk): {len(df):,} rows remain")

    # Build HDF5

    hdf5_indices  = []
    error_rows    = []
    n_text_only   = 0
    n_resampled   = 0
    idx           = 0

    # track errors
    failed_convos = set()
    
    DEBUG_CUR_ROW = 0
    with h5py.File(out_h5, "w") as hf:
        audio_grp = hf.create_group("audio")
        hf.attrs["source_df"]      = str(df_path)
        hf.attrs["audio_root_PSC"] = str(audio_root_PSC)
        hf.attrs["audio_root_ST"]  = str(audio_root_ST)

        if resample_rate is not None:
            hf.attrs["resample_rate"] = resample_rate

        for row_num, row in tqdm(df.iterrows(), total=DEBUG_MAX_ROW if DEBUG_MAX_ROW != -1 else len(df), desc="Writing audio"):
            if DEBUG_CUR_ROW == DEBUG_MAX_ROW: break
            DEBUG_CUR_ROW += 1

            record_type = str(row.get("record_type", "")).strip().lower()
            
            # Text-only row: no audio expected, skip silently 
            if record_type == "text_only":
                hdf5_indices.append(IDX_TEXT_ONLY)
                n_text_only += 1
                continue
                
            # missing file somehow slipped by preprocessing, skip whole convo
            if str(row.get("conv_id", "")) in failed_convos:
                hdf5_indices.append(IDX_ERROR)
                continue
            
            # check audio file sources
            source     = row.get("source", "")
            audio_root = SOURCE_ROOTS.get(source)

            if audio_root is None:
                print(f"  [ERROR] Unknown source '{source}' (row {row_num})")
                failed_convos.add(str(row.get("conv_id", "")))
                hdf5_indices.append(IDX_ERROR)
                error_rows.append(row_num)
                continue
            

            # check actual audio file
            rel_path  = str(row.get("relative_audio_path", "")).strip()
            full_path = audio_root / rel_path

            if not full_path.exists():
                
                print(f"  [ERROR] Missing file (row {row_num}): {full_path}")
                failed_convos.add(str(row.get("conv_id", "")))
                hdf5_indices.append(IDX_ERROR)
                error_rows.append(row_num)
                continue
            
            # load waveform
            try:
                waveform, sr = load_wav(full_path)
            except Exception as exc:
                print(f"  [ERROR] Unreadable file (row {row_num}): {full_path} -- {exc}")
                failed_convos.add(str(row.get("conv_id", "")))
                hdf5_indices.append(IDX_ERROR)
                error_rows.append(row_num)
                continue
            
            # perform resampling if passed and not already sampled at that rate
            if resample_rate is not None and sr != resample_rate:
                # upsampling from a lower SR doesn't recover lost information, so we treat it as bad data
                if sr < resample_rate:
                    print(f"  [ERROR] Native SR {sr} Hz is below target {resample_rate} Hz (row {row_num}): {full_path}")
                    failed_convos.add(str(row.get("conv_id", "")))
                    hdf5_indices.append(IDX_ERROR)
                    error_rows.append(row_num)
                    continue
 
                waveform = resample_waveform(waveform, sr, resample_rate)
                sr = resample_rate
                n_resampled += 1
            
            # build 'dataset' information for each waveform
            key = f"{idx:06d}"
            ds  = audio_grp.create_dataset(
                key,
                data=waveform,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            ds.attrs["sample_rate"]   = sr  # updated during resampling if that happened
            ds.attrs["duration_sec"]  = len(waveform) / sr
            ds.attrs["original_path"] = str(rel_path)
            ds.attrs["df_row"]        = int(row_num)

            for col in attr_cols_present:
                ds.attrs[col] = safe_attr(row[col])

            hdf5_indices.append(idx)
            idx += 1


        hf.attrs["total_audio_samples"] = idx
        hf.attrs["total_text_only"]     = n_text_only


    print(f"\nHDF5 written -> {out_h5}")
    print(f"  Audio samples stored : {idx:,}")
    print(f"  Text-only rows       : {n_text_only:,}  (hdf5_idx = {IDX_TEXT_ONLY})")
    print(f"  Unexpected errors    : {len(error_rows):,}  (hdf5_idx = {IDX_ERROR})")
    if error_rows:
        print(f"  Error row indices    : {error_rows}")
        print(f"  Conversations abandoned : {len(failed_convos):,}  (all turns marked hdf5_idx = {IDX_ERROR})")
    
    
    # Save metadata + index

    if DEBUG_MAX_ROW != -1:
        df = df[:DEBUG_MAX_ROW].copy()
    else:
        df = df.copy()

    df["hdf5_idx"] = hdf5_indices
    df.to_parquet(out_meta, index=False)

    print(f"\nMetadata written -> {out_meta}")
    print(f"  Join key : 'hdf5_idx'")
    print(f"    >= 0  -> audio present in HDF5")
    print(f"    {IDX_TEXT_ONLY}   -> text_only row (use transcription only)")
    print(f"    {IDX_ERROR}   -> audio expected but failed (investigate)")


def parse_args():
    p = argparse.ArgumentParser(description="Build HDF5 audio archive + Parquet metadata index")
    p.add_argument("--df",         default="../data/processed/merged_PSC_StyleTalk_CLEANED.parquet",  help="Path to input DataFrame (.csv or .parquet)")
    p.add_argument("--audio_root_PSC", default="../data/raw/paraspeechcaps/audio/expresso",  help="Root directory to paraspeechcaps that relative_audio_path is relative to")
    p.add_argument("--audio_root_ST", default="../data/raw/styletalk/audio",  help="Root directory to StyleTalk that relative_audio_path is relative to")
    p.add_argument("--out_h5",     default="../data/processed/merged_audio.h5",          help="Output HDF5 path")
    p.add_argument("--out_meta",   default="../data/processed/merged_metadata.parquet",   help="Output Parquet path")
    p.add_argument("--resample_rate",  default=None, type=int,
                   help="Resample all audio to this rate (Hz). Files with a native SR below this value will error.")

    p.add_argument("--SEED", type=int, help="Random seed for conversation sampling")
    p.add_argument("--DEBUG_MAX_ROW",   default=-1,   help="DEBUGGING: number of rows to use for smaller sample")
    p.add_argument("--DEBUG_MAX_TURNS", default=-1, help="DEBUGGING: only include utterances from conversations with turn_index <= this value")
    p.add_argument("--DEBUG_PERCENT_EXPRESSO", default=-1.0, type=float, help="Fraction of conversations from expresso source (0.0 to 1.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.df
          , args.audio_root_PSC
          , args.audio_root_ST
          , args.out_h5
          , args.out_meta
          , args.resample_rate
          , int(args.DEBUG_MAX_ROW)
          , int(args.DEBUG_MAX_TURNS)
          , float(args.DEBUG_PERCENT_EXPRESSO)
          , int(args.SEED))