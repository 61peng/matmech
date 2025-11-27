import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import argparse
from component.utils import generate_answer_local_api
from pathlib import Path
from tqdm import tqdm

system_message = "You are an expert in the field of materials science."

instruction = open("instruction/paper_type.md", "r", encoding="utf-8").read()

INPUT_TEMPLATE = "{instruction}\nHere is the material science paper in json list format:\n\n```json\n{paper_list}\n```"


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='判断文章类型')
    parser.add_argument('--model_name', type=str, default='Qwen3-Next-80B-A3B-Instruct', choices=['qwen-plus-latest', 'deepseek-chat'])
    parser.add_argument('--engine', type=str, default='api', choices=['local_api', "api"])
    parser.add_argument('--journal_name', type=str, default="Progress_in_Materials_Science")
    args = parser.parse_args()

    input_path = f"solved_pdf/{args.journal_name}"
    output_path = f"output_file/{args.journal_name}/raw_data/paper_type.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 已处理的 DOI 列表
    solved_list = []
    
    if os.path.exists(output_path):
        solved_list = [data['doi'] for data in [eval(line) for line in open(output_path, 'r', encoding='utf8')]]
    # 加载模型
    print(f"已处理{len(solved_list)}篇论文")

    f_o = open(output_path, "a+", encoding="utf-8")

    # 收集所有待处理的文件
    all_files = []
    file_dir = os.listdir(input_path)
    print(f"Total files to process: {len(file_dir)}")
    
    for doi in tqdm(file_dir):
        if doi in solved_list:
            continue
        full_paper_path = f"{input_path}/{doi}/auto/{doi}_content_list.json"
        if not os.path.exists(full_paper_path):
            continue
        paper_list = json.load(open(f"{input_path}/{doi}/auto/{doi}_content_list.json", "r", encoding="utf-8"))
        for idx, entry in enumerate(paper_list):
            entry.pop('page_idx', None)  # Remove 'page_idx' if it exists
            entry['block_index'] = idx  # Add index
        model_input = INPUT_TEMPLATE.format(instruction=instruction, paper_list=paper_list)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": model_input}
        ]

        output = generate_answer_local_api(messages, args.model_name)

        if output is not None:
            f_o.write(json.dumps({"doi": doi, "content": output}, ensure_ascii=False) + "\n")
            f_o.flush()

    f_o.close()
    print("Processing completed.")