import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
from tqdm import tqdm
from component.utils import generate_answer_api, generate_answer_local_api

system_message = "You are an expert in the field of materials science."

# 从instruction/mechanism_v2.md中读取
instruction = open("instruction/mechanism_v2.md", "r", encoding="utf-8").read()

INPUT_TEMPLATE = "{instruction}\nHere is the material science paper in json list format:\n\n```json\n{paper_list}\n```\nHere is the core knowledge dictionary:\n\n```json\n{core_information}\n```"

if __name__ == '__main__':
    # 首先读取已处理的文件
    parser = argparse.ArgumentParser(description='step3 arguments')
    parser.add_argument('--model_name', type=str, default='Qwen3-Next-80B-A3B-Instruct')
    parser.add_argument('--engine', type=str, default='local_api', choices=['local_api', "api"])
    parser.add_argument('--journal_name', type=str, default="Advanced_Energy_Materials")
    # parser.add_argument('--tetrahedron_path', type=str, default='output_file/nama/tetrahedron.json')
    # parser.add_argument('--image_path', type=str, default='output_file/nama/images_function.json')
    # parser.add_argument('--output_path', type=str, default='output_file/nama/mechanism_deepseekv3.jsonl')
    args = parser.parse_args()
    # 打印参数
    print(args)
    tetrahedron_path = f"output_file/{args.journal_name}/processed_data/tetrahedron.json"
    image_path = f"output_file/{args.journal_name}/processed_data/images_function.json"
    output_path = f"output_file/{args.journal_name}/raw_data/mechanism_{args.model_name}.jsonl"
    solved_path = f"output_file/{args.journal_name}/processed_data/mechanism.json"
    
    solved_list = []
    if os.path.exists(solved_path):
        solved_list = [doi for doi in json.load(open(solved_path, 'r', encoding='utf8'))]
    if os.path.exists(output_path):
        _solved_list = [data['doi'] for data in [eval(line) for line in open(output_path, 'r', encoding='utf8')]]
        solved_list.extend(_solved_list)
    
    # 去重
    solved_list = list(set(solved_list))
    f_o = open(output_path, "a+", encoding="utf-8")

    # 遍历文件夹下的所有文件
    with open(tetrahedron_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    with open(image_path, "r", encoding="utf-8") as f:
        image_dict = json.load(f)
    for doi, core_information in tqdm(json_dict.items()):
        if doi in solved_list:
            continue
        if core_information and not "logical chain" in core_information:
            continue
        if core_information and core_information["logical chain"]:
            try:
                json_file = f"solved_pdf/{args.journal_name}/{doi}/auto/{doi}_content_list.json"
                paper_list = json.load(open(json_file, "r", encoding="utf-8"))
                if len(paper_list) == 1:
                    continue
                for idx, entry in enumerate(paper_list):
                    entry.pop('page_idx', None)  # Remove 'page_idx' if it exists
                    entry.pop('bbox', None)  # Remove 'page_idx' if it exists
                    entry['block_index'] = idx  # Add index
                    if entry['type'] == 'image':
                        for item in image_dict[doi]:
                            if item['block_index'] == entry['block_index']:
                                entry['microscopic_image'] = item['microscopic_image']
                                entry['function'] = item['function']
                                entry['source'] = item['described_by']
                                break

                model_input = INPUT_TEMPLATE.format(instruction=instruction, paper_list=paper_list, core_information=core_information)
                # print(model_input)
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": model_input}
                ]
                if args.engine == "api":
                    output = generate_answer_api(messages, args.model_name, max_tokens=16392)
                else:
                    output = generate_answer_local_api(messages, args.model_name, max_tokens=16392)

                if not output == None:
                    content = {"doi": doi, "content": output}
                    f_o.write(json.dumps(content, ensure_ascii=False) + "\n")
                    f_o.flush()

            except Exception as e:
                print(f"Error processing {doi}: {e}")
                continue
    f_o.close()