import re
from datasets import load_dataset
import pandas as pd
import random

def initialize_expresso_df():
    
    # Load specific splits of the dataset

    train_base = load_dataset("ajd12342/paraspeechcaps", split="train_base")
    holdout = load_dataset("ajd12342/paraspeechcaps", split="holdout")
    dev = load_dataset("ajd12342/paraspeechcaps", split="dev")
    test = load_dataset("ajd12342/paraspeechcaps", split="test")

    dfs = [
        train_base.to_pandas(),
        holdout.to_pandas(),
        dev.to_pandas(),
        test.to_pandas(),
    ]

    # merge into one dataframe
    df = pd.concat(dfs, ignore_index=True)
    

    # Only want conversational, already segmented files
    df = df[df["source"] == "expresso"]
    
    
    df = df[df["relative_audio_path"].str.contains("conversational_vad_segmented")]

    # dedup: keep the row with the longer text_description list (throw out the basic tag style desc.)
    dupes_mask = df.duplicated(subset="relative_audio_path", keep=False)
    dupes = df[dupes_mask]
    if not dupes.empty:
        print(f"Found {len(dupes)} duplicate rows across {dupes['relative_audio_path'].nunique()} paths -- keeping longest text_description per path")
        
        # longer list wins; ties go to the first occurrence
        df["_desc_len"] = df["text_description"].apply(lambda x: len(x) if hasattr(x, '__len__') and not isinstance(x, str) and len(x) > 0 else 0)
        df = (
            df.sort_values("_desc_len", ascending=False)
              .drop_duplicates(subset="relative_audio_path", keep="first")
              .drop(columns="_desc_len")
              .reset_index(drop=True)
        )
    else:
        print("No duplicate paths found")

    # flatten: reduce text_description from list to its first string (keep only the most rich style description)
    df["text_description"] = df["text_description"].apply(
        lambda x: x[0] if hasattr(x, '__len__') and not isinstance(x, str) and len(x) > 0 else x
    )

    return df 


def find_missing_annotations(df):
    # check for files that are missing from conversations, skipping style folders that arent represented at all
    import os
    
    valid_styles = set(df['high_lvl_style'].unique())

    wav_files = []
    base_dir = "../data/raw/paraspeechcaps/audio/expresso/audio_48khz/conversational_vad_segmented"

    for root, dirs, files in os.walk(base_dir):
        # check if this directory is under a valid style folder
        style_match = re.search(r'conversational_vad_segmented/[^/]+/([\w-]+)', root.replace('\\', '/'))
        if style_match and style_match.group(1) not in valid_styles:
            continue
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    prefix = "../data/raw/paraspeechcaps/audio/expresso/"
    wav_files_rel = [f.replace('\\', '/').removeprefix(prefix) for f in wav_files]

    existing = set(df['relative_audio_path'].values)
    missing = [f for f in wav_files_rel if f not in existing]
    
    print(f"Total wav files found: {len(wav_files_rel)}")
    print(f"Missing from dataframe: {len(missing)}")
    
    return missing


def parse_start_time(filename):
    match = re.search(r'_segment_(\d+\.\d+)_', filename)
    return float(match.group(1)) if match else float('inf')

def parse_end_time(filename):
    match = re.search(r'_segment_\d+\.\d+_(\d+\.\d+)\.wav', filename)
    return float(match.group(1)) if match else float('inf')


def add_conversation_index(df_in):
    df = df_in.copy()
    
    # parse conversation info
    df['speakers'] = df['relative_audio_path'].str.extract(r'(ex\d+-ex\d+)')
    df['high_lvl_style'] = df['relative_audio_path'].str.extract(r'ex\d+-ex\d+/([\w-]+)/')
    df['conv_num'] = df['relative_audio_path'].str.extract(r'_(\d+)_channel')
    df['start_time'] = df['relative_audio_path'].apply(parse_start_time)
    df['end_time'] = df['relative_audio_path'].apply(parse_end_time)
    
    temp_columns_to_drop = ['speakers', 'high_lvl_style', 'conv_num', 'start_time', 'end_time']
    
    # Build conv_id from structured columns
    df['conv_id'] = df['speakers'] + '_' + df['high_lvl_style'] + '_' + df['conv_num']

    df = df.sort_values(['conv_id', 'start_time']).reset_index(drop=True)

    df['turn_index'] = df.groupby('conv_id').cumcount()
    df['prev_filename'] = df.groupby('conv_id')['relative_audio_path'].shift(1)
    
    # look for files that are missing annotaitons
    print("Looking for filenames that don't have annotations, theses could cause interruptions in conversations.")
    missing = find_missing_annotations(df)
    
    
    # separate conversations at the missing file, to keep conversations contiguous
    print("Splitting up conversations that have gaps.")
    df_split = df.copy()
    df_split['_is_split_point'] = False

    for f in missing:
        match = re.search(r'(ex\d+-ex\d+)_([\w-]+)_(\d+)_channel', f)
        if not match:
            continue
        conv_id = f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
        start = parse_start_time(f)
        if start is not None:
            # only mark the first row at or after the missing file's start time
            idx = df_split[(df_split['conv_id'] == conv_id) & (df_split['start_time'] >= start)].index
            if len(idx) > 0:
                df_split.loc[idx[0], '_is_split_point'] = True

    df_split['conv_id_split'] = df_split['conv_id'] + '_part' + df_split.groupby('conv_id')['_is_split_point'].transform(lambda x: x.cumsum().astype(str))

    df_split['turn_index'] = df_split.groupby('conv_id_split').cumcount()
    df_split['prev_filename'] = df_split.groupby('conv_id_split')['relative_audio_path'].shift(1)
    
    print(f"Original conversations: {df_split['conv_id'].nunique()}")
    print(f"Split conversations: {df_split['conv_id_split'].nunique()}")
    
    df_final = df_split.drop(columns=temp_columns_to_drop)
    
    return df_final


#
# StyleTalk incorporation
#

def parse_context_turns(context, diag_id):
    """Split context string into one dict per A/B turn."""
    
    # Matches single uppercase letter + optional spaces + colon: "A :", "B:", etc.
    SPEAKER_RE = re.compile(r'(?<!\w)([A-Z])\s*:')
    
    parts = SPEAKER_RE.split(context.strip())
    # re.split with a capturing group gives: ['', 'A', ' text ', 'B', ' text ', ...]
    turns = []
    turn_idx = 0
    i = 1
    while i + 1 < len(parts):
        speaker = parts[i].strip()
        text    = parts[i + 1].strip()
        if speaker and text:
            turns.append({
                'speaker_label': speaker,
                'transcription': text,
                'turn_index':    turn_idx,
                'conv_id':      diag_id,
            })
            turn_idx += 1
        i += 2
    return turns


def build_tags(emotion, speed, volume):
    """Combine emotion/speed/volume into tags, omitting 'normal' values. Must be in a list to match PSC"""
    parts = [
        str(v).strip()
        for v in [emotion, speed, volume]
        if pd.notna(v) and str(v).strip() != ''
    ]
    return parts if parts else pd.NA


def build_style_desc(emotion,speed,volume):
    """Builds simple sentence style description. Shuffle descriptions to better generalize"""
    parts = [
        f"in a {emotion} tone",
        f"at a {volume} volume",
        f"at a {speed} speed",
    ]
    random.shuffle(parts)
    desc = ", ".join(parts)
    return f"The speaker speaks {desc}."



def add_styletalk(psc_df, st_df):
    """Map styletalk columns onto psc columns for easier analysis and training"""
    
    MAIN_COLS = list(psc_df.columns)
    STYLETALK_EXTRA_COLS = ['emotion', 'volume']
    TYPE_COL = 'record_type'

    
    
    psc_df[TYPE_COL] = 'audio'
    new_rows = []

    for _, row in st_df.iterrows():
        diag_id = str(row['diag_id'])
        context = str(row.get('context', ''))
        
        curr_audio = row.get('curr_audio_id')

        # 1. Context turns → text_only rows (no audio)
        context_turns = []
        last_context_turn_id = pd.NA
        if context and context.lower() != 'nan':
            context_turns = parse_context_turns(context, diag_id)
            for turn in context_turns:
                r = {col: pd.NA for col in MAIN_COLS + STYLETALK_EXTRA_COLS}

                r[TYPE_COL]        = 'text_only'
                r['source']        = 'styletalk'
                r['conv_id']      = diag_id
                r['turn_index']    = turn['turn_index']
                
                # this is to match the history columns with the curr audio column.
                
                r['relative_audio_path']    = f"{curr_audio}_turn_{turn['turn_index']}"
                r['transcription'] = turn['transcription']
                r['speakerid']     = turn['speaker_label']   # 'A' or 'B'
                r['prev_filename'] = last_context_turn_id
                new_rows.append(r)

                # save for next turn
                last_context_turn_id = r['relative_audio_path']

        n_ctx = len(context_turns)  # offset so audio turns continue turn numbering

        # 2. curr audio row
        
        cur_speaker_label = pd.NA
        if pd.notna(curr_audio):
            r = {col: pd.NA for col in MAIN_COLS + STYLETALK_EXTRA_COLS}

            
            
            r[TYPE_COL]                 = 'audio'
            r['source']                 = 'styletalk'
            r['conv_id']               = diag_id
            r['turn_index']             = n_ctx
            r['relative_audio_path'] = curr_audio
            
            r['duration']               = row.get('curr_duration')
            r['utterance_pitch_mean']   = row.get('curr_utterance_pitch_mean')
            r['snr']                    = row.get('curr_snr')
            
            transcription = row.get('curr_text')
            
            
            match = re.match(r'^([AB])\s*:\s*', transcription)
            cur_speaker_label = match.group(1) if match else pd.NA
            if match:
                transcription = transcription[match.end():]
                
            
            r['transcription']          = transcription
            r['speakerid']              = cur_speaker_label
            r['speaking_rate']          = row.get('curr_speed')
            r['emotion']       = row.get('curr_emotion')
            r['volume']        = row.get('curr_volume')

            
            tags = build_tags(row.get('curr_emotion'),row.get('curr_speed'),row.get('curr_volume'))
            r['basic_tags']             = tags
            r['all_tags']               = tags
            
            r['text_description']       = build_style_desc(row.get('curr_emotion'),
                                            row.get('curr_speed'),
                                            row.get('curr_volume'))
            
            r['prev_filename'] = last_context_turn_id
            new_rows.append(r)

        # 3. res audio row
        res_audio = row.get('res_audio_id')
        
        if pd.notna(res_audio):
            r = {col: pd.NA for col in MAIN_COLS + STYLETALK_EXTRA_COLS}

            r[TYPE_COL]             = 'audio'
            r['source']             = 'styletalk'
            r['conv_id']           = diag_id
            r['turn_index']         = n_ctx + (1 if pd.notna(curr_audio) else 0)
            r['relative_audio_path']= res_audio
    
            r['duration']               = row.get('res_duration')
            r['utterance_pitch_mean']   = row.get('res_utterance_pitch_mean')
            r['snr']                    = row.get('res_snr')
            
            transcription = row.get('res_text')
            
            res_speaker_label = "A" if "A:" in transcription else "B" if "B:" in transcription else pd.NA
            if pd.notna(res_speaker_label):
                transcription = transcription.replace(res_speaker_label + ":", "")
            
            # no res speaker label in transcription but cur_speakerlabel had it (we know its flipped)
            elif pd.notna(cur_speaker_label):
                res_speaker_label = "A" if cur_speaker_label=="B" else "B"
                    
            r['transcription']          = transcription
            r['speakerid']              = res_speaker_label
    
            r['speaking_rate']      = row.get('res_speed')
            r['emotion']       = row.get('res_emotion')
            r['volume']        = row.get('res_volume')

            tags = build_tags(row.get('res_emotion'),row.get('res_speed'),row.get('res_volume'))
            r['basic_tags']             = tags
            r['all_tags']               = tags
            
            r['text_description']       = build_style_desc(row.get('res_emotion'),
                                            row.get('res_speed'),
                                            row.get('res_volume'))
            
            # prev_filename = the curr audio (the immediately preceding turn)
            r['prev_filename']      = curr_audio if pd.notna(curr_audio) else pd.NA
            new_rows.append(r)

    # merge
    new_df = pd.DataFrame(new_rows, columns=MAIN_COLS + [TYPE_COL] + STYLETALK_EXTRA_COLS)

    merged = pd.concat([psc_df, new_df], ignore_index=True)

    all_cols = MAIN_COLS + [TYPE_COL] + STYLETALK_EXTRA_COLS

    for c in merged.columns:
        if c not in all_cols:
            all_cols.append(c)
    merged = merged[all_cols]

    # cleaning leading or trailing quotes
    merged['text_description'] = merged['text_description'].str.strip('\'"')
    
    print(f"Done. {len(psc_df)} original + {len(new_df)} new = {len(merged)}")
    print(f"\nrecord_type breakdown:\n{merged[TYPE_COL].value_counts().to_string()}")
    
    return merged


def build_vocabulary(merged_df):
    """Build vocab from explicit category columns rather than all_tags."""
    import json

    expresso_df = merged_df[merged_df["source"] == "expresso"]
    st_audio_df = merged_df[(merged_df["source"] == "styletalk") & (merged_df["record_type"] == "audio")]

    expresso_columns = {
        "gender":        sorted(expresso_df["gender"].dropna().unique().tolist()),
        "accent":        sorted(expresso_df["accent"].dropna().unique().tolist()),
        "pitch":         sorted(expresso_df["pitch"].dropna().unique().tolist()),
        "speaking_rate": sorted(expresso_df["speaking_rate"].dropna().unique().tolist()),
        "noise": sorted(expresso_df["noise"].dropna().unique().tolist()),
        "intrinsic_tags": sorted(expresso_df["intrinsic_tags"].explode().dropna().unique().tolist()),
        "situational_tags": sorted(expresso_df["situational_tags"].explode().dropna().unique().tolist()),
    }
    styletalk_columns = {
        "emotion":       sorted(st_audio_df["emotion"].dropna().unique().tolist()),
        "volume":        sorted(st_audio_df["volume"].dropna().unique().tolist()),
        "speaking_rate": sorted(st_audio_df["speaking_rate"].dropna().unique().tolist()),
    }

    def _build_source_vocab(columns):
        # category → [tags]; a tag appearing in multiple categories is listed in each
        return {cat: vals for cat, vals in columns.items()}

    vocab = {
        "expresso":  _build_source_vocab(expresso_columns),
        "styletalk": _build_source_vocab(styletalk_columns),
    }



    out_path = "../eda/source_vocabularies.json"
    with open(out_path, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)

    print(f"Saved vocabulary to {out_path}")
    print(f"  expresso: {len(vocab['expresso'])} entries")
    print(f"  styletalk: {len(vocab['styletalk'])} entries")

    return vocab


def main():
    df = initialize_expresso_df()
    
    df = add_conversation_index(df)
    
    st_df = pd.read_parquet("../data/processed/styletalk_with_audio_stats.parquet")
    
    merged = add_styletalk(df, st_df)
    
    OUT_PQ = "../data_TEMP/merged_PSC_StyleTalk_CLEANED.parquet"
    merged.to_parquet(OUT_PQ, index=False)
    
    build_vocabulary(merged)

    
    
if __name__=="__main__":
    main()