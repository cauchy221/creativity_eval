import os
import random
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from trl.scripts.utils import TrlParser

# --- env ---
load_dotenv("/home/xliu1/creativity_eval/.env")
HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class ScriptArguments:
    dataset_path: str = field(default=None, metadata={"help": "Path to dataset JSON (with 'messages')"})
    model_id: str = field(default=None, metadata={"help": "Base model ID, e.g., meta-llama/Llama-3.1-70B"})
    max_seq_length: int = field(default=2048, metadata={"help": "Max sequence length"})

# --- map messages -> prompt/completion (strings) ---
def to_prompt_completion(ex):
    msgs = ex.get("messages", []) or []
    sys_txt   = next((m.get("content", "") for m in msgs if (m.get("role") or "").lower() == "system"), "")
    user      = next((m.get("content", "") for m in msgs if (m.get("role") or "").lower() == "user"), "")
    assistant = next((m.get("content", "") for m in msgs if (m.get("role") or "").lower() == "assistant"), "")

    # Explicit, stable boundary to avoid tokenizer prefix mismatches
    # - prompt ends with a separator and a cue
    # - completion starts cleanly (no leading space/BOS)
    prompt = (sys_txt.strip() + "\n\n" if sys_txt else "") + user.rstrip() + "\n\n### Response:\n"
    completion = assistant.lstrip()

    return {"prompt": prompt, "completion": completion}

def training_function(script_args: ScriptArguments, training_args: SFTConfig):
    # --- dataset ---
    ds = load_dataset("json", data_files=script_args.dataset_path, split="train")
    ds = ds.map(
        to_prompt_completion,
        remove_columns=[c for c in ds.column_names if c not in {"prompt", "completion"}],
    )

    # --- tokenizer (no Llama-instruct chat template on base) ---
    tok = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True, token=HF_TOKEN)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    # Avoid BOS being inconsistently added on one side of the pair
    if hasattr(tok, "add_bos_token"):
        tok.add_bos_token = False

    # quick sanity print
    with training_args.main_process_first(desc="samples"):
        for i in random.sample(range(min(len(ds), 5)), k=min(2, len(ds))):
            print("-" * 60)
            print("PROMPT:\n", ds[i]["prompt"][:400])
            print("---\nCOMPLETION:\n", ds[i]["completion"][:400])

    # --- QLoRA 4-bit ---
    torch_dtype = torch.bfloat16
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quant_cfg,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=not training_args.gradient_checkpointing,
        token=HF_TOKEN,
    )
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- LoRA ---
    peft_cfg = LoraConfig(
        r=16, lora_alpha=8, lora_dropout=0.05,
        bias="none", target_modules="all-linear", task_type="CAUSAL_LM",
    )

    # --- trainer (TRL 0.23: prompt/completion -> completion-only loss by default) ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,          # SFTConfig (contains max_seq_length, packing, etc.)
        train_dataset=ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )

    # --- train & save ---
    ckpt = getattr(training_args, "resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=ckpt)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    set_seed(training_args.seed)
    training_function(script_args, training_args)
