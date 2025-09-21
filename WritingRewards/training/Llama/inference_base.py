# inference_base_like_instruct.py
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

def save_partial(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic rename on POSIX

def is_peft_adapter(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))

def read_base_from_adapter_dir(adapter_dir: str):
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (
                data.get("base_model_name_or_path")
                or data.get("base_model_name")
                or data.get("model_name_or_path")
            )
        except Exception:
            pass
    return None

def build_prompt_from_messages(messages):
    """
    Recreate the training-time prompt for base SFT:
      [optional system]\n\n<user>\n\n### Response:\n
    """
    sys_txt = ""
    user_txt = ""
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "system":
            sys_txt = (m.get("content") or "").strip()
        elif role == "user" and not user_txt:
            user_txt = (m.get("content") or "").strip()
    return (sys_txt + "\n\n" if sys_txt else "") + user_txt + "\n\n### Response:\n"

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--examples_json", type=str, required=True)
    args.add_argument("--messages_json", type=str, required=True)
    args.add_argument("--output_json", type=str, required=True)
    args.add_argument("--temperature", type=float, default=0.2)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--autosave_every", type=int, default=5)
    parsed = args.parse_args()

    # seeds
    torch.manual_seed(parsed.seed)
    random.seed(parsed.seed)
    np.random.seed(parsed.seed)

    # model + tokenizer (mirror your instruct code style)
    if is_peft_adapter(parsed.model_path):
        # LoRA adapter dir
        model = AutoPeftModelForCausalLM.from_pretrained(
            parsed.model_path,
            token=token,
            torch_dtype=torch.float16,
            quantization_config={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
            device_map="auto",
        )
        # Prefer the base id from adapter_config for the tokenizer
        base_id = read_base_from_adapter_dir(parsed.model_path) or parsed.model_path
        tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True, token=token)
    else:
        # Plain (merged) model dir or hub id
        model = AutoModelForCausalLM.from_pretrained(
            parsed.model_path,
            token=token,
            torch_dtype=torch.float16,
            quantization_config={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(parsed.model_path, use_fast=True, token=token)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    # keep BOS out to mirror training prompt
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False

    # load inputs
    with open(parsed.examples_json, "r", encoding="utf-8") as f:
        examples = json.load(f)
    eval_dataset = load_dataset("json", data_files=parsed.messages_json, split="train")

    predictions = []
    try:
        for idx, example in enumerate(tqdm(examples, total=len(examples), desc="Examples")):
            # system + first user (same as training)
            messages = eval_dataset[idx]["messages"][:2]
            prompt = build_prompt_from_messages(messages)

            word_count = example.get("word_count", 200)
            max_token_count = int(word_count * 4 / 3) + 50

            # tokenize once; mirror your instruct path’s approach
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            prompt_len = input_ids.size(-1)

            generations = []
            chunk = 1  # same parallel sampling batch size you use for instruct
            total = 100
            for i in range(0, total, chunk):
                cur = min(chunk, total - i)
                batch_inputs = input_ids.repeat(cur, 1)

                outputs = model.generate(
                    batch_inputs,
                    max_new_tokens=max_token_count,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=parsed.temperature,
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

            output_example = dict(example)
            output_example["generations"] = generations
            predictions.append(output_example)

            if (idx + 1) % parsed.autosave_every == 0:
                save_partial(parsed.output_json, predictions)

            # break

    finally:
        save_partial(parsed.output_json, predictions)
        print(f"Saved {len(predictions)} examples → {parsed.output_json}")
