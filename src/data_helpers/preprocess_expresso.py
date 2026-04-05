"""
Adapted from Diwan et al.'s apply_expresso_vad.py, fixed bugs, added logging and multithreading.
Combined with normalization step to run the full pipeline in one pass.
"""
import argparse
from pathlib import Path
import pydub
import sys
import os
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm


def load_vad_segments(vad_file):
    vad_segments = {}  # {filename: {channel: [(start, end)]}}

    with open(vad_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or 'longform' in line:
                continue

            filename_with_channel, times_str = line.split('\t')
            filename, channel = filename_with_channel.split('/')

            if filename not in vad_segments:
                vad_segments[filename] = {}

            if channel in vad_segments[filename]:
                print(f"Warning: Duplicate channel {channel} for {filename}", file=sys.stderr)
                continue

            vad_segments[filename][channel] = [
                (float(start), float(end))
                for part in times_str.strip().split(') ')
                if part and (start := part.replace('(', '').replace(')', '').split(', ')[0])
                and (end := part.replace('(', '').replace(')', '').split(', ')[1])
            ]

    return vad_segments


# lock so concurrent workers don't interleave their stderr warnings
_print_lock = threading.Lock()

def warn(msg):
    with _print_lock:
        print(msg, file=sys.stderr)


def process_audio_file(wav_file, vad_segments, output_dir):
    filename = wav_file.stem
    if filename not in vad_segments:
        warn(f"Warning: VAD segments not found for {filename}")
        return 0, []

    # load before mkdir so we don't create empty dirs for broken files
    try:
        audio = pydub.AudioSegment.from_wav(str(wav_file))  # str() avoids Path issues on Windows
    except Exception as e:
        warn(f"Error loading {wav_file}: {e}")
        return 0, []

    if len(audio) == 0:
        warn(f"Warning: Empty audio in {wav_file}, skipping")
        return 0, []

    if audio.channels < 2:
        warn(f"Warning: Expected stereo, got {audio.channels} channel(s) in {wav_file}, skipping")
        return 0, []

    output_dir.mkdir(parents=True, exist_ok=True)

    mono_channels = audio.split_to_mono()  # split once, reuse
    channels = {
        'channel1': mono_channels[0],
        'channel2': mono_channels[1]
    }

    segments_written = 0
    output_paths = []
    for channel_name, audio_channel in channels.items():
        if channel_name not in vad_segments[filename]:
            warn(f"Warning: No VAD segments for {filename}/{channel_name}")
            continue

        for start, end in vad_segments[filename][channel_name]:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            # guard against VAD timestamps exceeding actual audio length
            if start_ms >= len(audio_channel):
                warn(f"Warning: Segment {start}-{end}s out of range for {filename}/{channel_name}, skipping")
                continue

            segment = audio_channel[start_ms:end_ms]
            segment_path = output_dir / f"{filename}_{channel_name}_segment_{start}_{end}.wav"
            segment.export(segment_path, format="wav")
            segments_written += 1
            output_paths.append(segment_path)

    return segments_written, output_paths


def normalize_file(input_path: Path, input_root: Path, output_root: Path):
    rel_path = input_path.relative_to(input_root)
    output_path = output_root / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return

    cmd = ["sox", str(input_path), str(output_path), "norm", "-0.1"]
    subprocess.run(cmd, check=True)


def run_vad_segmentation(args, vad_segments, input_dir, segmented_dir):
    wav_files = list(input_dir.glob('**/*.wav'))
    total_files = len(wav_files)
    print(f"Found {total_files} file(s), segmenting with {args.workers} threads...")

    all_segment_paths = []
    total_segments = 0

    def submit(wav_file):
        relative_path = wav_file.relative_to(input_dir)
        output_subdir = segmented_dir / relative_path.parent
        return process_audio_file(wav_file, vad_segments, output_subdir)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(submit, f): f for f in wav_files}
        for future in tqdm(as_completed(futures), total=total_files, desc=f"Segmenting ({args.workers} workers)", unit="file"):
            wav_file = futures[future]
            try:
                segments_written, paths = future.result()
                total_segments += segments_written
                all_segment_paths.extend(paths)
            except Exception as e:
                warn(f"Error processing {wav_file}: {e}")

    print(f"Segmentation done. {total_segments} segment(s) written to {segmented_dir}")
    return all_segment_paths


def run_normalization(segment_paths, segmented_dir, normalized_dir, workers):
    print(f"\nNormalizing {len(segment_paths)} segment(s) using {workers} workers...")

    # follow symlinks to match bash's `find -L` behavior — also catches any
    # pre-existing files if segmentation was partially run before
    wav_files = [
        Path(dirpath) / fname
        for dirpath, _, filenames in os.walk(segmented_dir, followlinks=True)
        for fname in filenames
        if fname.endswith(".wav")
    ]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(normalize_file, wav, segmented_dir, normalized_dir) for wav in wav_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Normalizing", unit="file"):
            pass

    print(f"Normalization done. Output at {normalized_dir}")


def main():
    parser = argparse.ArgumentParser(description="Segment and normalize Expresso conversational audio")
    parser.add_argument('--expresso_root', default=Path("../data/raw/paraspeechcaps/audio/expresso"),type=Path, help='Root directory of Expresso dataset')
    parser.add_argument('--output_root', default=Path("../data/processed/paraspeechcaps/audio/expresso/audio_48khz"), type=Path, help='Root directory for processed output')
    parser.add_argument('--workers', type=int, default=8, help='Worker count for both stages (default: 8)')
    parser.add_argument('--skip-segmentation', action='store_true', help='Skip VAD segmentation, normalize existing segments only')
    args = parser.parse_args()

    vad_file = args.expresso_root / "VAD_segments.txt"
    input_dir = args.expresso_root / "audio_48khz" / "conversational"

    # keep intermediate segments separate so they're resumable/inspectable
    # must match structure of original
    segmented_dir = args.output_root  / "segmented_temp"
    normalized_dir = args.output_root  / "conversational_vad_segmented"  # this must be the final folder to match structure in dataset relative paths

    if not args.skip_segmentation:
        if not vad_file.is_file():
            print(f"Error: VAD segments file not found at {vad_file}", file=sys.stderr)
            sys.exit(1)
        if not input_dir.is_dir():
            print(f"Error: Input directory not found at {input_dir}", file=sys.stderr)
            sys.exit(1)

        vad_segments = load_vad_segments(vad_file)
        segment_paths = run_vad_segmentation(args, vad_segments, input_dir, segmented_dir)
    else:
        print("Skipping segmentation, scanning existing segments...")
        segment_paths = list(segmented_dir.glob('**/*.wav'))

    if not segment_paths:
        print("No segments to normalize, exiting.", file=sys.stderr)
        sys.exit(1)

    run_normalization(segment_paths, segmented_dir, normalized_dir, args.workers)
    print(f"\nAll done. Final output: {normalized_dir}")


if __name__ == '__main__':
    main()