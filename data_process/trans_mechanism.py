import re
import json

journal_name = "Advanced_Energy_Materials"
pattern = r'```json\s*(\[\s*\{.*?\}\s*\])\s*```'
mechanism = {}
valid_mechanism_count = 0
for line in open(f"output_file/{journal_name}/raw_data/mechanism_Qwen3-Next-80B-A3B-Instruct.jsonl", "r", encoding='utf8'):
    info = json.loads(line)
    text = info["content"]
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            mechanism_json_str = match.group(1)
            mechanism_json = json.loads(mechanism_json_str)

        except json.JSONDecodeError as e:
            print(f"{info["doi"]}\tJSONDecodeError: {e}")
            mechanism_json = []
        # print(mechanism_json)
    else:
        print(info["doi"])
        mechanism_json = []
    

    if len(mechanism_json) > 0:
        valid_mechanism_count += 1
        mechanism[info["doi"]] = mechanism_json

print(f"Valid mechanism count: {valid_mechanism_count}")

with open(f"output_file/{journal_name}/processed_data/mechanism.json", "w", encoding='utf8') as f:
    json.dump(mechanism, f, indent=4, ensure_ascii=False)