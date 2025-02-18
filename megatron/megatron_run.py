# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os
import json
from itertools import product
import argparse

sys.path.append("..")
from prompts import (
    make_direct_output_prompt,
    make_cot_output_prompt,
    make_direct_input_prompt,
    make_cot_input_prompt,
)

from megatron_prompt import (
    extract_answer_direct_input,
    extract_answer_cot_input,
    extract_answer_direct_output,
    extract_answer_cot_output,
)
from megatron_prompt import batch_prompt_megatron


def run_megatron(root, model, mode, batch_size,
                 cot, max_tokens, temperature, n_samples, base_url):
    # Load evaluation data.
    dataset_path = os.path.join("..", "data", "cruxeval.jsonl")
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()]

    # Prepare raw prompt tuples.
    if mode == "input":
        raw_prompts = [(data["code"], data["output"]) for data in dataset]
    else:
        raw_prompts = [(data["code"], data["input"]) for data in dataset]

    # # Set maximum tokens based on whether chain-of-thought is enabled.
    # Specify outside
    # if cot: 
    #     max_tokens = 1000
    # else: 
    #     max_tokens = 100

    # Select the appropriate prompt formatting function.
    prompt_fn = {
        (True, "input"): make_cot_input_prompt,
        (True, "output"): make_cot_output_prompt,
        (False, "input"): make_direct_input_prompt,
        (False, "output"): make_direct_output_prompt,
    }[(cot, mode)]

    # Select the extraction function to process raw model generations.
    #TODO: What about the extraction function for huggingface models?
    # if the same, why write this logic under openai?
    extraction_fn = {
        (True, "input"): extract_answer_cot_input,
        (True, "output"): extract_answer_cot_output,
        (False, "input"): extract_answer_direct_input,
        (False, "output"): extract_answer_direct_output,
    }[(cot, mode)]

    # Generate model completions (10 per prompt).
    formatted_prompts = [prompt_fn(p) for p in raw_prompts]
    raw_outputs = batch_prompt_megatron(
        formatted_prompts,
        temperature,
        batch_size,
        n=n_samples,
        max_gen_toks=max_tokens,
        base_url=base_url,
        stop=["[/ANSWER]"],  # ['r"\\Z"']  # not sure about this
        stop_type="string",  # "regex"
    )

    # Post-process outputs: extract the desired answer from each generation.
    outputs = [[extraction_fn(gen) for gen in gen_list] for gen_list in raw_outputs]


    # Save the results.
    save_path = get_save_dir(root, mode, model, cot, temperature)
    outputs_dict = {f"sample_{i}": outputs[i] for i in range(len(outputs))}
    with open(save_path, "w") as f:
        json.dump(outputs_dict, f)
    print(save_path)
    return outputs

def get_save_dir(root, mode, model, cot, temperature):
    """
    Create and return the directory path where the generated outputs will be saved.
    """
    if cot:
        base_dir = os.path.join(root, f"{model}+cot_temp{temperature}_{mode}")
    else:
        base_dir = os.path.join(root, f"{model}_temp{temperature}_{mode}")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "generations.json")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single configuration of Megatron")
    parser.add_argument(
        "--root",
        type=str,
        default="..",
        help="Root dir to save the generation results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="megatron",
        help="Model identifier (default: 'megatron')"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="input",
        choices=["input", "output"],
        help="Mode to use (default: 'input')"
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=80,
        help="Batch size"
    )
    parser.add_argument(
        "--cot",
        type=str2bool,
        default=False,
        help="Chain-of-thought flag (default: False). Accepts true/false values."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="Max number of generated tokens"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature value (default: 0.2)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples (default to be 10 per CruxEval leaderboard)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        required=True,
        help="Base URL for the API or service"
    )

    args = parser.parse_args()
    # run_megatron(root, model, mode, cot, max_tokens, temperature, n_samples, base_url):
    print(args)
    run_megatron(args.root,
                 args.model,
                 args.mode,
                 args.bs,
                 args.cot,
                 args.max_tokens,
                 args.temperature,
                 args.n_samples,
                 args.base_url)

    # models = ["megatron"]  # Example model identifiers.
    # modes = ["input", "output"]
    # cots = [False, True]
    # temperatures = [0.2]  # for pass@1, leaderboard setup
    # base_url = 

    # for model, mode, cot, temperature in product(models, modes, cots, temperatures):
        # run_megatron(model, mode, cot, max_tokens, temperature, base_url)
        # break  # Remove or comment out this line to run all combinations.
