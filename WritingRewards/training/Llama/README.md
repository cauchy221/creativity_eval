# Usage

## Prepare environment
- Make sure you are under `WritingRewards/training/Llama` all the time.
- Install packages after creating a new env: `pip install -r requirements.txt`.
- Create a `.env` file with your `HF_TOKEN` and `HF_HOME`. Put the `.env` file under project root: `creativity_eval/.env`.

# Prepare data
- Put raw author data under `data/`. For example:
```txt
data/
└── Sally_Rooney/
    └── output_Sally_Rooney_-_Normal_People.json
    └── output_Sally_Rooney_-_Conversations_with_Friends.json
```
- Convert raw data into messages format with `convert_messages.py`. The resulted data folder should look like:
```txt
data/
└── Sally_Rooney/
    └── output_Sally_Rooney_-_Normal_People.json
    └── output_Sally_Rooney_-_Conversations_with_Friends.json
    └── normal.json
    └── conv.json
```

## Finetuning
- We use `sft.py` for instruct-model finetuning, and `sft_base.py` for base-model finetuning. Remember to update the path in `load_dotenv()` before starting.
- For instruct-model finetuning, refer to `llama_3_70b_fsdp_qlora.yaml` for setting up parameters and configs. Choose the `model_id`, `dataset_path`, `output_dir` accordingly. Start finetuning with:
```bash
CUDA_VISIBLE_DEVICES={your_gpu_id} torchrun --nproc_per_node={num_gpu} sft.py --config llama_3_70b_fsdp_qlora.yaml
```
- For base-model finetuning, refer to `sft_base.yaml`for setting up parameters and configs. Choose the `model_id`, `dataset_path`, `output_dir` accordingly. Start finetuning with:
```bash
CUDA_VISIBLE_DEVICES={your_gpu_id} torchrun --nproc_per_node={num_gpu} sft_base.py --config sft_base.yaml
```
- The finetuned model checkpoints will be saved under your chosen `output_dir` in the yaml file.


## Inference
- We use `inference2.py` for instruct-model inference, and `inference_base.py` for base-model inference. Remember to update the path in `load_dotenv()` before starting.
- Instruct-model inference example command. Please update parameters accordingly:
```bash
CUDA_VISIBLE_DEVICES={your_gpu_id} python inference2.py --model_path "meta-llama/Llama-3.3-70B-Instruct" --examples_json "./data/Sally_Rooney/output_Sally_Rooney_-_Normal_People.json"   --messages_json "./data/Sally_Rooney/normal.json"   --output_json "./data/Sally_Rooney/pt-instruct-temp=10-test-normal.json"   --temperature 1.0
```
```bash
CUDA_VISIBLE_DEVICES={your_gpu_id} python inference2.py --model_path "./checkpoint/Sally_Rooney-conv-instruct/checkpoint-102" --examples_json "./data/Sally_Rooney/output_Sally_Rooney_-_Normal_People.json"   --messages_json "./data/Sally_Rooney/normal.json"   --output_json "./data/Sally_Rooney/ft-instruct-temp=10-conv-epoch=1-test-normal.json"   --temperature 1.0
```
- Base-model inference example command. Please update parameters accordingly:
```bash
CUDA_VISIBLE_DEVICES={your_gpu_id} python inference_base.py --model_path "meta-llama/Llama-3.1-70B" --examples_json "./data/Sally_Rooney/output_Sally_Rooney_-_Normal_People.json"   --messages_json "./data/Sally_Rooney/normal.json"   --output_json "./data/Sally_Rooney/pt-base-temp=10-test-normal.json"   --temperature 1.0
```
```bash
CUDA_VISIBLE_DEVICES={your_gpu_id} python inference_base.py --model_path "./checkpoint/Sally_Rooney-conv-base/checkpoint-102" --examples_json "./data/Sally_Rooney/output_Sally_Rooney_-_Normal_People.json"   --messages_json "./data/Sally_Rooney/normal.json"   --output_json "./data/Sally_Rooney/ft-base-temp=10-conv-epoch=1-test-normal.json"   --temperature 1.0
```
- Based on your GPU RAM, adjust `chunk (default=50)` in the inference code. This number represents the number of examples the model inference at the same time.