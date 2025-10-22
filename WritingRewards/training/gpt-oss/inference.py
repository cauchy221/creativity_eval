import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from datasets import load_dataset
import os
from tqdm import tqdm
import torch


def save_partial(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def post_process_response(response):
    final_response = response
    
    if "<|channel|>final<|message|>" in response:
        final_response = response.split("<|channel|>final<|message|>")[1]
    elif response.startswith("final"):
        final_response = response[5:]
    
    # Remove end tokens
    if "<|return|>" in final_response:
        final_response = final_response.split("<|return|>")[0]
    
    return final_response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test messages json file")
    parser.add_argument("--raw_book", type=str, required=True, help="Path to the raw book json file")
    parser.add_argument("--generation_output", type=str, required=True, help="Path to the generation output json file")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=25, help="Generations per batch")
    parser.add_argument("--autosave_every", type=int, default=5, help="Save every N examples")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--num_generations", type=int, default=100, help="Number of generations per example")
    args = parser.parse_args()

    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # load base model
    model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs).cuda()

    # load finetuned adapter
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model = model.merge_and_unload()
    model.eval()

    # load raw book data
    with open(args.raw_book, "r", encoding="utf-8") as f:
        raw_book_examples = json.load(f)

    # load test data
    test_dataset = load_dataset("json", data_files=args.test_data, split="train")

    # inference
    outputs = []
    start_idx = 0
    if args.resume and os.path.exists(args.generation_output):
        try:
            with open(args.generation_output, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                outputs = existing
                start_idx = len(existing)
                print(f"[resume] Found {start_idx} completed examples in {args.generation_output}.")
        except Exception as e:
            print(f"[resume] Could not parse existing output ({e}). Starting from scratch.")

    num_generations = args.num_generations
    batch_size = args.batch_size

    try:
        for idx in tqdm(range(start_idx, len(raw_book_examples)), 
                       desc="Processing examples", 
                       initial=start_idx, 
                       total=len(raw_book_examples)):

            raw_book_example = raw_book_examples[idx]
            test_example = test_dataset[idx]

            word_count = raw_book_example['word_count']
            max_token_count = int(word_count * 4 / 3) + 50

            messages = test_example['messages'][:2]  # exclude the last message (the assistant message)
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                # you can set reasoning effort here; default is "medium"
                # reasoning_effort=["low", "medium", "high"]
            ).to(model.device)
            prompt_len = input_ids.size(-1)

            gen_kwargs = {
                "max_new_tokens": max_token_count,
                "do_sample": True,
                "temperature": args.temperature,
            }

            generations = []
            for batch_start in range(0, num_generations, batch_size):
                current_batch_size = min(batch_size, num_generations - batch_start)
                batch_input_ids = input_ids.repeat(current_batch_size, 1)

                with torch.no_grad():
                    output_ids = model.generate(batch_input_ids, **gen_kwargs)

                generated_tokens = output_ids[:, prompt_len:]
                responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                for response in responses:
                    final_response = post_process_response(response)
                    
                    generations.append({
                        "generation_num": len(generations) + 1,
                        "generated_text": final_response,
                    })

            output_example = raw_book_example.copy()
            output_example["generations"] = generations
            outputs.append(output_example)

            if (idx + 1) % args.autosave_every == 0:
                save_partial(args.generation_output, outputs)
                tqdm.write(f"[AUTOSAVE] Saved {len(outputs)} examples at index {idx + 1}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving progress before exit...")

    finally:
        save_partial(args.generation_output, outputs)
        print(f"\n[FINAL SAVE] Saved {len(outputs)} examples to {args.generation_output}")


if __name__ == "__main__":
    main()