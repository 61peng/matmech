import json
import re

def extract_json(content):
    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError:
            json_obj = None
    else:
        print("content", content)
        json_obj = None  # 表示 None 或解析失败
    return json_obj

tetrahedron = {}
journal_name = "Rare_Metals"

input_file = f"output_file/{journal_name}/raw_data/category.jsonl"
output_file = f"output_file/{journal_name}/processed_data/category.json"

for line in open(input_file, "r", encoding='utf8'):
    info = json.loads(line)
    text = info["content"]

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = extract_json(text)

        
    tetrahedron[info["doi"]] = result

with open(output_file, "w", encoding='utf8') as f:
    json.dump(tetrahedron, f, indent=4, ensure_ascii=False)