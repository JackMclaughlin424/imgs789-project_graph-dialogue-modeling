import wave
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import parselmouth


def get_wav_pitch_mean(y: np.ndarray, sr: int) -> float:
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=500)
        values = pitch.selected_array['frequency']
        voiced = values[values > 0]
        return float(np.mean(voiced)) if len(voiced) > 0 else np.nan
    except Exception as e:
        print(f"pitch failed: {e}", flush=True)
        return np.nan


def get_wav_snr(y: np.ndarray) -> float:
    """Blind SNR estimate using RMS energy - pure numpy."""
    try:
        frame_length = 512
        hop = 256
        frames = np.array([
            y[i:i + frame_length]
            for i in range(0, len(y) - frame_length, hop)
        ])
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        threshold = np.percentile(rms, 30)
        noise_power  = np.mean(rms[rms <= threshold] ** 2)
        signal_power = np.mean(rms[rms >  threshold] ** 2)
        if noise_power <= 0:
            return np.nan
        return float(10 * np.log10(signal_power / noise_power))
    except:
        return np.nan


def extract_features(filepath: Path) -> dict:
    y, sr = sf.read(str(filepath), dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    return {
        'duration':             float(len(y) / sr),
        'utterance_pitch_mean': get_wav_pitch_mean(y, sr),
        'snr':                  get_wav_snr(y),
    }


def load_styletalk(base_dir: str = "../data/raw/styletalk") -> pd.DataFrame:
    splits = ["train", "eval"]
    dfs = []
    for split in splits:
        path = f"{base_dir}/annotations/{split}.csv"
        dfs.append(pd.read_csv(path, sep=",", encoding="utf-8", na_values=["NA", ""]))
    return pd.concat(dfs, ignore_index=True)


def enrich_with_audio_features(
    df: pd.DataFrame,
    audio_root: str = "../data/raw/styletalk/audio",
    n_jobs: int = 8,
) -> pd.DataFrame:
    result_df = df.copy()
    audio_cols = {'curr': 'curr_audio_id', 'res': 'res_audio_id'}

    for prefix, col in audio_cols.items():
        paths = [Path(audio_root) / p for p in df[col]]
        print(f"Starting '{prefix}': {len(paths)} files...", flush=True)

        results_map = {}
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_idx = {executor.submit(extract_features, p): i
                             for i, p in enumerate(paths)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_map[idx] = future.result()
                    if idx % 50 == 0:
                        print(f"  {prefix}: {idx}/{len(paths)} done", flush=True)
                except Exception as e:
                    print(f"  Failed idx {idx} ({paths[idx]}): {e}", flush=True)
                    results_map[idx] = {'duration': np.nan, 'utterance_pitch_mean': np.nan, 'snr': np.nan}

        results = [results_map[i] for i in range(len(paths))]
        features_df = pd.DataFrame(results, index=df.index).rename(columns={
            'duration':             f'{prefix}_duration',
            'utterance_pitch_mean': f'{prefix}_utterance_pitch_mean',
            'snr':                  f'{prefix}_snr',
        })
        result_df = pd.concat([result_df, features_df], axis=1)
        print(f"'{prefix}' done.", flush=True)

    return result_df


if __name__ == "__main__":
    st_df = load_styletalk()

    test_path = Path("../data/raw/styletalk/audio") / st_df["curr_audio_id"].iloc[0]
    print(f"Testing single file: {test_path}", flush=True)
    print(extract_features(test_path), flush=True)
    print("Single file OK", flush=True)

    st_df = enrich_with_audio_features(st_df)
    print(st_df[["curr_duration", "curr_utterance_pitch_mean", "curr_snr",
                  "res_duration",  "res_utterance_pitch_mean",  "res_snr"]].describe())

    st_df.to_parquet("../data/processed/styletalk_with_audio_stats.parquet", index=False)