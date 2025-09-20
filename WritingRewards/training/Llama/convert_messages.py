import json

SYSTEM_PROMPT = "You are an AI assistant who has knowledge about creative writing."

raw_data = "./data/Sally_Rooney/output_Sally_Rooney_-_Normal_People.json"
output_data = "./data/Sally_Rooney/normal.json"

with open(raw_data, "r", encoding="utf-8") as f:
    examples = json.load(f)

with open(output_data, "w", encoding="utf-8") as f:
    for example in examples:
        message = {
            "messages": [
                {
                    "content": SYSTEM_PROMPT,
                    "role": "system"
                },
                {
                    "content": example["instruction"],
                    "role": "user"
                },
                {
                    "content": example["paragraph_text"],
                    "role": "assistant"
                }
            ]
        }
        f.write(json.dumps(message, ensure_ascii=False) + "\n")
