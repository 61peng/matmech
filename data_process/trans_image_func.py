import re
import json
pattern = r'```json\s*(\[\s*\{.*?\}\s*\])\s*```'
images = {}
for line in open("output_file/Advanced_Composites_and_Hybrid_Materials/raw_data/images_function_Qwen3-Next-80B-A3B-Instruct.jsonl", "r", encoding='utf8'):
    info = json.loads(line)
    text = info["content"]
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            mechanism_json_str = match.group(1)
            mechanism_json = json.loads(mechanism_json_str)  # 解析 JSON
            # print(mechanism_json)
        except json.JSONDecodeError as e:
            print(info["doi"])
            print(f"JSONDecodeError: {e}")
            mechanism_json = []
    else:
        mechanism_json = []
    images[info["doi"]] = mechanism_json

with open("output_file/Advanced_Composites_and_Hybrid_Materials/processed_data/images_function.json", "w", encoding='utf8') as f:
    json.dump(images, f, indent=4, ensure_ascii=False)