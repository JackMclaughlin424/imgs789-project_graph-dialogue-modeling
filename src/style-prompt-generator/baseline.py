"""
Baseline: few-shot style prompt generation using TinyLlama via HuggingFace transformers.
Samples 3 random chains from ConvoStyleDataset to use as in-context examples,
then predicts the style for a query chain's final utterance.
"""

import os
import random
import textwrap
import argparse
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ConvoStyleDataset import ConvoStyleDataset

from train_helpers import compute_bertscore, compute_meteor

# same repo used by StylePromptGenerator -- keep these in sync
LLM_REPO = "meta-llama/Llama-3.2-3B-Instruct"  # or whichever you choose
LLM_DIM  = 3072  # must match model's hidden_size



# W&B config -- set USE_WANDB=False or unset WANDB_API_KEY to disable
USE_WANDB      = True
WANDB_PROJECT  = "style-prompt-gen"
WANDB_ENTITY   = "jdm8943-rochester-institute-of-technology"
WANDB_RUN_NAME = "baseline-llama3B-fewshot"


def load_dataset(h5_path: str, meta_path: str, num_turns: int, max_len_sec: int) -> ConvoStyleDataset:
    return ConvoStyleDataset(
        h5_path=h5_path,
        meta_path=meta_path,
        meta_columns=["transcription", "text_description", "speakerid"],
        num_turns=num_turns,
        max_len_sec=max_len_sec,
    )


def chain_to_text(chain: list, include_last_style: bool = True) -> str:
    lines = []
    for i, utt in enumerate(chain):
        speaker = utt.get("speakerid", f"Speaker{i}")
        text    = utt.get("transcription", "").strip()
        is_last = i == len(chain) - 1

        lines.append(f"[Turn {i+1}] {speaker}: \"{text}\"")
        if is_last:
            lines.append(f"  -> Style: ???")

    return "\n".join(lines)



def build_few_shot_example(chain: list) -> str:
    lines = []
    for i, utt in enumerate(chain):
        speaker = utt.get("speakerid", f"Speaker{i}")
        text    = utt.get("transcription", "").strip()
        is_last = i == len(chain) - 1

        lines.append(f"[Turn {i+1}] {speaker}: \"{text}\"")
        if is_last:
            style = (utt.get("text_description") or "unknown").strip()
            lines.append(f"  -> Style: {style}")

    return "\n".join(lines)



def build_system_prompt(few_shot_chains: list[list]) -> str:
    task_description = textwrap.dedent("""\
        You are a prompt assistant for a conversational, instruction-guided text-to-speech system.
        Your task is to predict an appropriate natural-language style description (style prompt) for the NEXT utterance
        to be synthesized, given the dialogue history and the script (transcription) of that next utterance.

        The style description should reflect how the speech should sound -- covering aspects like emotion,
        speaking rate, vocal quality, and energy level -- and should be consistent with the speaker's
        established style, and be appropriate to the conversational flow. Be concise: 1-3 sentences is ideal.

        Each turn shows the speaker and their transcription.
        For the final turn, predict the style and output it as "Style: <your prediction>".
        Respond with only the style description for the final turn, nothing else.
    """)


    examples_block = "\n\n---\n\n".join(
        f"Example {i+1}:\n{build_few_shot_example(chain)}"
        for i, chain in enumerate(few_shot_chains)
    )

    return f"{task_description}\nHere are {len(few_shot_chains)} examples:\n\n{examples_block}"


def build_user_prompt(query_chain: list) -> str:
    dialogue_block = chain_to_text(query_chain, include_last_style=False)
    return (
        "Given the dialogue history and the script for the final turn, "
        "predict the style description.\n\n"
        f"{dialogue_block}\n"
        "  -> Style:"  # exact match to few-shot example format; model completes here
    )




# wandb helpers

def wandb_init(cfg: dict):
    if not USE_WANDB:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed -- skipping W&B tracking.")
        return None
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("WANDB_API_KEY not set -- skipping W&B tracking.")
        return None
    wandb.login(key=api_key, relogin=False)
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            config=cfg,
        )
        print(f"W&B run: {run.url}")
        return run
    except Exception as e:
        print(f"W&B init failed ({e}) -- continuing without tracking.")
        return None


def wandb_log(metrics: dict, run):
    if run is not None:
        run.log(metrics)


def wandb_finish(run):
    if run is not None:
        run.finish()




def load_llm(device: str, repo: str = LLM_REPO):
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model



def query_llm(tokenizer, model, system_prompt, user_prompt, device, max_new_tokens) -> str:
    # Raw completion format — lets few-shot examples condition the pattern directly
    # rather than the chat template overriding with instruction-following behavior
    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
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


    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()



def main(
    h5_path: str,
    meta_path: str,
    num_turns: int = 5,
    num_few_shot: int = 3,
    num_eval_samples: int = 50,
    max_new_tokens: int = 80,
    max_len_sec: int = 15,
    seed: int = 42,
    llm_repo: str = LLM_REPO,
):

    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    ds = load_dataset(h5_path, meta_path, num_turns, max_len_sec)
    print(f"  {len(ds)} chains available")

    # filter out text-only rows w/ no text_description
    all_indices = [
        i for i, chain in enumerate(ds._chains)
        if all(
            chain_row.get("text_description") not in (None, "")
            and not (isinstance(chain_row.get("text_description"), float) and np.isnan(chain_row.get("text_description")))
            for chain_row in chain
        )
    ]

    query_indices = random.sample(all_indices, min(num_eval_samples, len(all_indices)))

    print(f"\n{len(query_indices)} query chains sampled for evaluation.")

    print(f"\nLoading LLM on {device}...")
    tokenizer, model = load_llm(device, repo=llm_repo)

    wandb_run = wandb_init({
        "model":            llm_repo,
        "num_turns":        num_turns,
        "num_few_shot":     num_few_shot,
        "num_eval_samples": num_eval_samples,
        "max_new_tokens":   max_new_tokens,
        "max_len_sec":      max_len_sec,
        "seed":             seed,
    })


    all_preds, all_refs = [], []

    for idx, qi in enumerate(query_indices):
        few_shot_pool   = [i for i in all_indices if i != qi]
        few_shot_chains = [ds[i] for i in random.sample(few_shot_pool, num_few_shot)]
        system_prompt   = build_system_prompt(few_shot_chains)

        query_chain  = ds[qi]
        user_prompt  = build_user_prompt(query_chain)
        prediction   = query_llm(tokenizer, model, system_prompt, user_prompt, device, max_new_tokens)

        ground_truth = query_chain[-1].get("text_description", "").strip()

        all_preds.append(prediction)
        all_refs.append(ground_truth)

        print(f"\n[{idx+1}/{len(query_indices)}]")
        print(f"  Predicted : {prediction}")
        print(f"  Reference : {ground_truth}")

    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    bs_metrics  = compute_bertscore(all_preds, all_refs, device=device)
    met_metrics = compute_meteor(all_preds, all_refs)

    print(f"BERTScore F1  : {bs_metrics['bertscore_f1_mean']:.4f} (±{bs_metrics['bertscore_f1_std']:.4f})")
    print(f"BERTScore P   : {bs_metrics['bertscore_precision_mean']:.4f} (±{bs_metrics['bertscore_precision_std']:.4f})")
    print(f"BERTScore R   : {bs_metrics['bertscore_recall_mean']:.4f} (±{bs_metrics['bertscore_recall_std']:.4f})")
    print(f"METEOR        : {met_metrics['meteor_mean']:.4f} (±{met_metrics['meteor_std']:.4f})")

    wandb_log({**bs_metrics, **met_metrics}, wandb_run)
    wandb_finish(wandb_run)

    return {**bs_metrics, **met_metrics}


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot LLM baseline for style prompt prediction.")
    p.add_argument("--h5_path",          required=True,              help="Path to merged_audio.h5")
    p.add_argument("--meta_path",        required=True,              help="Path to merged_metadata.parquet")
    p.add_argument("--num_turns",        type=int,   default=5,      help="Conversation chain length (default: 5)")
    p.add_argument("--num_few_shot",     type=int,   default=3,      help="Number of in-context examples (default: 3)")
    p.add_argument("--num_eval_samples", type=int,   default=50,     help="Number of query chains to evaluate (default: 50)")
    p.add_argument("--max_new_tokens",   type=int,   default=80,     help="Max tokens to generate per prediction (default: 80)")
    p.add_argument("--max_len_sec",      type=int,   default=15,     help="Max audio length in seconds (default: 15)")
    p.add_argument("--llm_repo", type=str, default=LLM_REPO, help="HuggingFace repo for the LLM (default: Llama 3.2 3B)")

    p.add_argument("--seed",             type=int,   default=42,     help="Random seed (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
