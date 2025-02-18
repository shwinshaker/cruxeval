import time
import math
import requests


def unflatten(flat_list, sizes):
    """
    Convert a flattened list into a nested list based on the provided sizes.
    """
    nested_list = []
    index = 0
    for size in sizes:
        nested_list.append(flat_list[index:index+size])
        index += size
    return nested_list

def batch_prompt_megatron(prompts, temperature, batch_size, n, max_gen_toks, base_url, stop, stop_type):
    do_sample = True if temperature > 0 else False
    total_batches = math.ceil(len(prompts) / batch_size)

    # Process the prompts in chunks of `batch_size`
    all_outputs = []
    for batch_index, i in enumerate(range(0, len(prompts), batch_size), start=1):
        current_chunk = prompts[i:i+batch_size]
        # Build the batch arguments for this chunk only.
        batch_args = []
        # import pdb; pdb.set_trace()
        for prompt in current_chunk:
            for _ in range(n):
                batch_args.append((
                    prompt,
                    {
                        "max_gen_toks": max_gen_toks,
                        "until": stop,
                        "until_type": stop_type,
                        "do_sample": do_sample,
                        "temperature": temperature,
                    }
                ))
        
        # Optionally print batch_args for debugging.
        # print("Batch args:", batch_args)

        print(f"Sending batch [{batch_index}/{total_batches}] with {len(batch_args)} requests..")
        batch_start = time.time()
        response = requests.post(
            base_url,
            json={"args": batch_args},
            timeout=300,
        )
        batch_outputs = response.json()["text"]
        # Unflatten the flat list of outputs into a nested list for the current chunk.
        chunk_outputs = unflatten(batch_outputs, [n] * len(current_chunk))
        # import pdb; pdb.set_trace()
        batch_time = time.time() - batch_start
        remaining_batches = total_batches - batch_index
        eta = batch_time * remaining_batches
        print(f"Batch [{batch_index}/{total_batches}] completed in {batch_time:.2f} seconds. ETA for remaining batches: {eta/60:.2f} mins.")

        all_outputs.extend(chunk_outputs)
    
    return all_outputs


# copied from openai_prompt.py, because import that file would require OPENAI_API_KEY
def extract_answer_direct_output(gen):
    if "==" in gen:
        gen = gen.split("==")[1]
    return gen.strip()

def extract_answer_direct_input(gen):
    if "==" in gen:
        gen = gen.split("==")[0].strip()
    if "assert f" in gen:
        gen = "f" + gen.split("assert f")[1].strip()
    return gen.strip()

def extract_answer_cot_input(gen):
    if "[ANSWER]" in gen:
        gen = gen.split("[ANSWER]")[1].strip()
        if "==" in gen:
            gen = gen.split("==")[0]
        if "assert f" in gen:
            gen = "f" + gen.split("assert f")[1].strip()
        return gen.strip()
    else:
        return gen.split('\n')[-1].strip()

def extract_answer_cot_output(gen):
    if "[ANSWER]" in gen:
        gen = gen.split("[ANSWER]")[1].strip()
        if "==" in gen:
            gen = gen.split("==")[1]
        return gen.strip()
    else:
        return gen.split('\n')[-1].strip()