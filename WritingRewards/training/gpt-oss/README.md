# GPT-OSS Instruction
We follow the example here: https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers


## Env setup
Create a new env:
```bash
conda create -n gpt-oss python=3.11
conda activate gpt-oss
```

Install packages:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0"
pip install tensorboard
pip install kernels
pip install deepspeed
```


## Model training
We are finetuning gpt-oss model with lora. Here are the main files:
```txt
sft_lora.py: main lora finetuning script
sft_config.yaml: setting finetuning parameters
ds_config.json: deepspeed config file
```

Please follow these steps:
1. Setup the finetuning parameters (eg. bs, epochs) in `sft_config.yaml`, depending on your compute
2. You may also need to set up the lora config in `sft_lora.py`. Specifically, you can add lora to a subset of projection layers as shown below. If you have enough compute, you can also add adapters to all layers by changing it to `target_parameters=all_targets`
```python
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
```
3. Depending on the compute, setup deepspeed in `ds_config.json` (I'm using stage 3 CPU-offloading here). You can change to stage 2, 1, or even no off loading if you have enough compute (if you don't need deepspeed, simply remove the last line `deepspeed: "ds_config.json"` in `sft_config.yaml`)
4. Put the training data file under `/gpt-oss/data/`. It should be in the same format as `combined_exclude_normal_messages.json`
5. Create a foler `/gpt-oss/checkpoint/` for saving your finetuned adapters
6. Run finetuning with
```bash
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc-per-node=2 sft_lora.py \
--train_data "./data/combined_exclude_normal_messages.json" \
--model_id openai/gpt-oss-20b \
--output_dir "./checkpoint/gpt-oss-20b-lora" \
--sft_config_yaml sft_config.yaml
```

Tips:
- It's not enough to just finetune 1 epoch (at least for the 20b model). The generation quality is bad
- gpt-oss is a reasoning model, and our training data does not contain any CoT details. If the finetuning is weak (eg. only one epoch with limited data), the model might generate some thinking content in the final response


## Model inference
Please follow these steps:
1. Put the test data file (in messages format), and the raw book file under `/gpt-oss/data/`
2. Since gpt-oss is a reasoning model, you may update its reasoning effort during inference in `inference.py`
```python
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    # you can set reasoning effort here; default is "medium"
    # reasoning_effort=["low", "medium", "high"]
).to(model.device)
```
3. Start inference with the following command
```bash
CUDA_VISIBLE_DEVICES=3 python inference.py \
--model_id openai/gpt-oss-20b \
--adapter_dir "./checkpoint/gpt-oss-20b-lora" \  # finetuned lora adapter
--test_data "./data/normal.json" \  # test file in messages format
--raw_book "./data/output_Sally_Rooney_-_Normal_People.json" \  # raw book file
--generation_output "./data/ft-temp=10-(n-1)-test-normal.json" \  # generation output
--temperature 1.0 \
--batch_size 25 \  # batch size for batch generation; adjust based on compute
--autosave_every 5  # auto save the temporary output file every X examples
```