import argparse
import torch
from transformers import AutoTokenizer, Mxfp4Config, AutoModelForCausalLM, HfArgumentParser
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from trl import SFTTrainer, SFTConfig


@dataclass
class ScriptArgs:
    train_data: str = field(metadata={"help": "Path to training data (JSON with 'messages')."})
    model_id: str = field(metadata={"help": "Model ID (e.g., openai/gpt-oss-20b)."})
    output_dir: str = field(metadata={"help": "Where to save the fine-tuned model."})
    sft_config_yaml: str = field(default="sft_config.yaml", metadata={"help": "Path to SFTConfig YAML."})


def ensure_final_channel(msgs):
    """
    For gpt-oss Harmony format, it's recommended to ensure the final channel in assistant messages.
    Final channel messages are the ones shown to the end user.
    This is because we can also include thinking/analysis messages in training data.
    """
    out = []
    for msg in msgs:
        if msg.get("role") == "assistant" and "channel" not in msg:
            msg = dict(msg)
            msg["channel"] = "final"
        out.append(msg)
    return out


def prepare_dataset(tokenizer, train_data):
    # load raw training data
    ds = load_dataset("json", data_files=train_data, split="train")
    # print(ds[0])

    def to_text(ex):
        msgs = ensure_final_channel(ex["messages"])
        return {
            "text": tokenizer.apply_chat_template(
                msgs, 
                tokenize=False
            )
        }

    ds = ds.map(to_text, remove_columns=["messages"])
    return ds


def prepare_model(model_id):
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        # device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return model


def find_all_targets(model):
    # find how many layers
    num_layers = len(model.model.layers)
    print(f"Number of layers: {num_layers}")
    targets = []
    for i in range(num_layers):
        targets.append(f"{i}.mlp.experts.gate_up_proj")
        targets.append(f"{i}.mlp.experts.down_proj")
    return targets


def prepare_peft_model(base_model):
    all_targets = find_all_targets(base_model)
    # print(all_targets)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",  # attention layers
        target_parameters=[  # a subset of projection layers within the experts
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
        # target_parameters=all_targets  # if have enough memory, we can use all targets
    )
    peft_model = get_peft_model(base_model, peft_config)
    return peft_model


def main():
    parser = HfArgumentParser((ScriptArgs,))
    (script_args,) = parser.parse_args_into_dataclasses()

    sft_parser = HfArgumentParser(SFTConfig)
    (training_args,) = sft_parser.parse_yaml_file(script_args.sft_config_yaml)
    training_args.output_dir = script_args.output_dir


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

    # prepare dataset
    train_dataset = prepare_dataset(tokenizer, script_args.train_data)

    # load model
    base_model = prepare_model(script_args.model_id)

    # debug: check a response
    # messages = [
    #     {"role": "user", "content": "¿Cuál es el capital de Australia?"},
    # ]

    # input_ids = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     return_tensors="pt",
    # ).to(base_model.device)

    # output_ids = base_model.generate(input_ids, max_new_tokens=512)
    # response = tokenizer.batch_decode(output_ids)[0]
    # print(response)

    # load lora model
    peft_model = prepare_peft_model(base_model)
    peft_model.print_trainable_parameters()

    # start training
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(script_args.output_dir)


if __name__ == "__main__":
    main()