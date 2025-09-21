import argparse
import json
import os
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

load_dotenv("/home/xliu1/creativity_eval/.env")
token = os.getenv("HF_TOKEN")

# --- ADDED: safe partial save helper ---
def save_partial(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic rename on POSIX

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    args.add_argument("--examples_json", type=str, default="./data/Sally_Rooney/output_Sally_Rooney_-_Normal_People.json")
    args.add_argument("--messages_json", type=str, default="./data/Sally_Rooney/normal.json")
    args.add_argument("--output_json", type=str, default="./data/Sally_Rooney/pt-temp=02-test-normal.json")
    args.add_argument("--temperature", type=float, default=0.2)
    args.add_argument("--seed", type=int, default=42)
    # ADDED: let autosave cadence be adjustable (default 5)
    args.add_argument("--autosave_every", type=int, default=5)
    args = args.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # base model
    if args.model_path == "meta-llama/Llama-3.3-70B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            token=token,
            torch_dtype=torch.float16,
            quantization_config={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, token=token)
    # peft model
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            token=token,
            torch_dtype=torch.float16,
            quantization_config={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    with open(args.examples_json, "r", encoding="utf-8") as f:
        examples = json.load(f)

    eval_dataset = load_dataset("json", data_files=args.messages_json, split="train")
    predictions = []

    try:
        for idx, example in enumerate(tqdm(examples, total=len(examples), desc="Examples")):
            messages = eval_dataset[idx]["messages"][:2]
            word_count = example['word_count']
            max_token_count = int(word_count * 4 / 3) + 50

            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            prompt_len = input_ids.size(-1)

            generations = []
            chunk = 50  # your chosen batch of parallel samples per example

            for i in range(0, 100, chunk):
                cur = min(chunk, 100 - i)
                batch_inputs = input_ids.repeat(cur, 1)

                outputs = model.generate(
                    batch_inputs,
                    max_new_tokens=max_token_count,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=args.temperature,
                )

                for j in range(outputs.size(0)):
                    gen_tokens = outputs[j, prompt_len:]
                    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    generations.append({
                        "generation_num": len(generations) + 1,
                        "generation": text
                    })
                # print(generations)
                # break

            output_example = example.copy()
            output_example["generations"] = generations
            predictions.append(output_example)

            # --- ADDED: autosave every N examples ---
            if (idx + 1) % args.autosave_every == 0:
                save_partial(args.output_json, predictions)
            # break

    finally:
        # --- ADDED: always save final state (even if crash/interrupt) ---
        save_partial(args.output_json, predictions)
        print(f"Saved {len(predictions)} examples → {args.output_json}")