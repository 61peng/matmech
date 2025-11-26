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
for line in open("output_file/Journal_of_Magnesium_and_Alloys/raw_data/tetrahedron_Qwen3-Next-80B-A3B-Instruct.jsonl", "r", encoding='utf8'):
    info = json.loads(line)
    text = info["content"]

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = extract_json(text)

        
    tetrahedron[info["doi"]] = result

with open("output_file/Journal_of_Magnesium_and_Alloys/processed_data/tetrahedron.json", "w", encoding='utf8') as f:
    json.dump(tetrahedron, f, indent=4, ensure_ascii=False)