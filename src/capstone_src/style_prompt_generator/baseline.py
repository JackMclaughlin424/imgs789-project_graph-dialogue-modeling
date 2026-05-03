"""
baseline.py  --h5_path ...  --meta_path ...  --sweep_values sweep_values.json
             [--sweep_id <ID>] [--count N] [options]

Runs a W&B sweep of the few-shot LLM baseline for style prompt prediction.
Multiple agents can join the same sweep via --sweep_id.
"""


import os
import json
import random
import gc
import textwrap
import argparse
from copy import deepcopy
import numpy as np
import pandas as pd

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset.ConvoStyleDataset import ConvoStyleDataset
from model.train_helpers import compute_bertscore, compute_meteor, compute_chrf, compute_rouge, compute_tag_f1

from tqdm import tqdm


LLM_REPO       = "meta-llama/Llama-3.2-3B-Instruct"
LLM_DIM  = 3072  # must match model's hidden_size

WANDB_PROJECT  = "style-prompt-gen"
WANDB_ENTITY   = "jdm8943-rochester-institute-of-technology"



def load_dataset(h5_path: str, meta_path: str, num_turns: int, max_len_sec: int) -> ConvoStyleDataset:
    return ConvoStyleDataset(
        h5_path=h5_path,
        meta_path=meta_path,
        meta_columns=["transcription", "text_description", "speakerid", "conv_id"],
        num_turns=num_turns,
        max_len_sec=max_len_sec,
    )



def chain_to_text(chain: list) -> str:
    lines = []
    for i, utt in enumerate(chain):
        text    = utt.get("transcription", "").strip()
        is_last = i == len(chain) - 1

        lines.append(f"[Turn {i+1}]")
        lines.append(f"  Transcription: \"{text}\"")
        if is_last:
            lines.append(f"  Speaking style:")
        else:
            style = (utt.get("text_description") if isinstance(utt.get("text_description"), str) else None) or "unknown"
            style = style.strip()
            lines.append(f"  Speaking style: {style}")

    return "\n".join(lines)






def build_few_shot_example(chain: list) -> str:
    lines = []
    for i, utt in enumerate(chain):
        text  = utt.get("transcription", "").strip()
        style = (utt.get("text_description") if isinstance(utt.get("text_description"), str) else None) or "unknown"
        style = style.strip()


        lines.append(f"[Turn {i+1}]")
        lines.append(f"  Transcription: \"{text}\"")
        lines.append(f"  Speaking style: {style}")

    return "\n".join(lines)





def build_system_prompt(few_shot_chains: list[list]) -> str:
    task_description = textwrap.dedent("""\
        You are a prompt assistant for a conversational, instruction-guided text-to-speech system.
        Your task is to predict an appropriate natural-language style description (style prompt) for the NEXT utterance
        to be synthesized, given the dialogue history and the script (transcription) of that next utterance.

        The style description should reflect how the speech should sound -- covering aspects like emotion,
        speaking rate, vocal quality, and energy level -- and should be consistent with the speaker's
        established style, and be appropriate to the conversational flow. Be concise: 1-3 sentences is ideal.

        Each turn shows the transcription, and (for all turns except the final query turn) the speaking style.
        For the final turn, only the transcription is provided.
        Predict the style for that final turn and output it as "Style: <your prediction>".
        Respond with only the content of your style prediction for the final turn, nothing else.

    """)


    examples_block = "\n\n---\n\n".join(
        f"Example {i+1}:\n{build_few_shot_example(chain)}"
        for i, chain in enumerate(few_shot_chains)
    )

    return f"{task_description}\nHere are {len(few_shot_chains)} examples:\n\n{examples_block}"


def build_user_prompt(query_chain: list) -> str:
    dialogue_block = chain_to_text(query_chain)
    return (
        "Given the dialogue history and the script for the final turn, "
        "predict only the final style description.\n\n"
        f"{dialogue_block}\n"
    )




def load_llm(device: str, repo: str = LLM_REPO):
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # causal LMs must left-pad for batched generation
    return tokenizer, model




def batch_query_llm(
    tokenizer, model, full_prompts: list[str], device: str,
    max_new_tokens: int, batch_size: int = 8
) -> list[str]:
    results = []
    for start in tqdm(range(0, len(full_prompts), batch_size), desc="Inference batches"):

        batch = full_prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for seq in output_ids:
            generated = seq[prompt_len:]
            results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
        
        del inputs, output_ids
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return results




def run_baseline(cfg: dict, run) -> dict:
    seed                = cfg["seed"]
    h5_path             = cfg["h5_path"]
    meta_path           = cfg["meta_path"]
    num_turns           = cfg["num_turns"]
    num_few_shot        = cfg["num_few_shot"]
    num_eval_samples    = cfg["num_eval_samples"]
    max_new_tokens      = cfg["max_new_tokens"]
    inference_batch_size = cfg["inference_batch_size"]
    max_len_sec         = cfg["max_len_sec"]
    llm_repo            = cfg.get("llm_repo", LLM_REPO)
    data_source         = cfg.get("data_source", "both")
    output_path         = f"baseline_outputs_{run.id}.json" if run else cfg.get("output_path", "baseline_outputs.json")

    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    ds = load_dataset(h5_path, meta_path, num_turns, max_len_sec)
    print(f"  {len(ds)} chains available")

    meta_df = pd.read_parquet(meta_path)

    if data_source != "both":
        other_source = "styletalk" if data_source == "expresso" else "expresso"
        source_ids   = set(meta_df[meta_df["source"] == data_source]["conv_id"].unique())
        other_ids    = np.array(meta_df[meta_df["source"] == other_source]["conv_id"].unique())
        rng_test     = np.random.default_rng(seed)
        rng_test.shuffle(other_ids)
        test_ids     = set(other_ids)
        print(
            f"  {len(source_ids)} {data_source} conv_ids for few-shot pool  |  "
            f"{len(test_ids)} {other_source} conv_ids for test (cross-dataset)"
        )
    else:
        source_ids   = None
        all_conv_ids = np.array(meta_df["conv_id"].unique())
        rng_split    = np.random.default_rng(seed)
        shuffled_all = all_conv_ids.copy()
        rng_split.shuffle(shuffled_all)
        n_test   = max(1, int(len(shuffled_all) * 0.10))
        test_ids = set(shuffled_all[:n_test])
        print(f"  Test split: {len(test_ids)} conv_ids (10% held-out, seed={seed})")

    def has_valid_styles(chain):
        # only the final turn's style is required (it's the prediction target)
        last = chain[-1]
        val  = last.get("text_description")
        return (
            val not in (None, "")
            and not (isinstance(val, float) and np.isnan(val))
        )


    test_indices = [
        i for i, chain in enumerate(ds._chains)
        if chain[-1].get("conv_id") in test_ids and has_valid_styles(chain)
    ]
    train_indices = [
        i for i, chain in enumerate(ds._chains)
        if chain[-1].get("conv_id") not in test_ids
        and (source_ids is None or chain[-1].get("conv_id") in source_ids)
        and has_valid_styles(chain)
    ]

    query_indices = random.sample(test_indices, min(num_eval_samples, len(test_indices)))
    print(f"\n{len(query_indices)} query chains sampled for evaluation.")
    print(f"  Few-shot pool size: {len(train_indices)} train chains")

    print(f"\nLoading LLM on {device}...")
    tokenizer, model = load_llm(device, repo=llm_repo)

    full_prompts, ground_truths, meta = [], [], []
    for qi in tqdm(query_indices, desc="Building prompts"):
        few_shot_chains = [ds[i] for i in random.sample(train_indices, num_few_shot)]
        system_prompt   = build_system_prompt(few_shot_chains)
        query_chain     = ds[qi]
        user_prompt     = build_user_prompt(query_chain)
        full_prompts.append(f"{system_prompt}\n\n---\n\n{user_prompt}")
        ground_truths.append(query_chain[-1].get("text_description", "").strip())
        meta.append({"index": qi, "system_prompt": system_prompt, "user_prompt": user_prompt})

    print(f"\nRunning batched inference over {len(full_prompts)} prompts...")
    predictions = batch_query_llm(tokenizer, model, full_prompts, device, max_new_tokens, batch_size=inference_batch_size)

    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    all_preds, all_refs = [], []
    records = []
    for idx, (pred, gt, m) in enumerate(zip(predictions, ground_truths, meta)):
        records.append({**m, "prediction": pred, "ground_truth": gt})
        all_preds.append(pred)
        all_refs.append(gt)
        print(f"\n[{idx+1}/{len(query_indices)}]  Predicted: {pred}\n  Reference: {gt}")

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} inference records to {output_path}")

    bs_metrics    = compute_bertscore(all_preds, all_refs, device=device)
    met_metrics   = compute_meteor(all_preds, all_refs)
    chrf_metrics  = compute_chrf(all_preds, all_refs)
    rouge_metrics = compute_rouge(all_preds, all_refs)
    tag_metrics   = compute_tag_f1(all_preds, all_refs)

    print(f"BERTScore F1 : {bs_metrics['bertscore_f1_mean']:.4f}")
    print(f"METEOR       : {met_metrics['meteor_mean']:.4f}")
    print(f"ChrF++       : {chrf_metrics['chrf_mean']:.4f}")
    print(f"ROUGE        : {rouge_metrics['rouge_mean']:.4f}")
    print(f"Tag F1       : {tag_metrics['tag_f1_overall']:.4f}")

    results = {**bs_metrics, **met_metrics, **chrf_metrics, **rouge_metrics, **tag_metrics}
    if run:
        run.log(results)


    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results




def _make_sweep_fn(base_cfg: dict):
    def sweep_fn():
        gc.collect()
        run = wandb.init(settings=wandb.Settings(console="off"))
        cfg = deepcopy(base_cfg)
        for key, val in run.config.items():
            cfg[key] = val
        print(f"Run config: {json.dumps(cfg, indent=2, default=str)}")
        results = run_baseline(cfg, run)
        run.summary.update(results)
        run.finish()
    return sweep_fn


def create_sweep(sweep_values, project: str = WANDB_PROJECT, entity: str = WANDB_ENTITY) -> str:
    """Create a W&B sweep and return the sweep ID.

    sweep_values: path to a sweep JSON file, or an already-loaded dict.
    """
    if isinstance(sweep_values, str):
        with open(sweep_values) as f:
            sweep_values = json.load(f)
    sweep_id = wandb.sweep(deepcopy(sweep_values), project=project, entity=entity)
    print(f"Created sweep: {sweep_id}")
    return sweep_id


def run_agent(
    base_cfg: dict,
    sweep_id: str,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    count: int = None,
):
    """Run a sweep agent against an existing sweep (blocking).

    Intended for notebook use — call from separate cells or threads.
    base_cfg should contain all fixed params (h5_path, meta_path, seed, etc.).
    """
    sweep_fn = _make_sweep_fn(base_cfg)
    wandb.agent(sweep_id, function=sweep_fn, count=count, project=project, entity=entity)



def parse_args():
    p = argparse.ArgumentParser(description="W&B sweep of the few-shot LLM baseline.")
    # required fixed params
    p.add_argument("--h5_path",              required=True,              help="Path to merged_audio.h5")
    p.add_argument("--meta_path",            required=True,              help="Path to merged_metadata.parquet")
    p.add_argument("--sweep_values",         required=True,              help="W&B sweep search-space JSON")
    # sweep control
    p.add_argument("--sweep_id",             default=None,               help="Join an existing sweep instead of creating one")
    p.add_argument("--count",                type=int,   default=None,   help="Max trials this agent will run (default: unlimited)")
    # fixed hyperparams (not swept unless added to sweep_values.json)
    p.add_argument("--num_turns",            type=int,   default=5)
    p.add_argument("--num_few_shot",         type=int,   default=25)
    p.add_argument("--num_eval_samples",     type=int,   default=200)
    p.add_argument("--max_new_tokens",       type=int,   default=80)
    p.add_argument("--inference_batch_size", type=int,   default=8)
    p.add_argument("--max_len_sec",          type=int,   default=15)
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--llm_repo",             type=str,   default=LLM_REPO)
    p.add_argument("--data_source",          type=str,   default="both",  choices=["both", "styletalk", "expresso"])
    # wandb
    p.add_argument("--wandb_project",        type=str,   default=WANDB_PROJECT)
    p.add_argument("--wandb_entity",         type=str,   default=WANDB_ENTITY)
    return p.parse_args()



def main():
    args = parse_args()
    cfg  = vars(args)

    sweep_values_path = cfg.pop("sweep_values")
    sweep_id_arg      = cfg.pop("sweep_id")
    count             = cfg.pop("count")
    project           = cfg.pop("wandb_project")
    entity            = cfg.pop("wandb_entity")

    with open(sweep_values_path) as f:
        sweep_config = json.load(f)

    sweep_fn = _make_sweep_fn(cfg)

    if sweep_id_arg:
        sweep_id = sweep_id_arg
        print(f"Joining existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        print(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_fn, count=count, project=project, entity=entity)



if __name__ == "__main__":
    main()
