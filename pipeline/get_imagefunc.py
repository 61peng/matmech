import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from component.utils import generate_answer_api, generate_answer_local_api
import argparse
from tqdm import tqdm



system_message = "You are an expert in the field of materials science."

instruction = open("instruction/image_function.md", "r", encoding="utf-8").read()


INPUT_TEMPLATE = "{instruction}\n\nHere is the material science paper in json list format:\n\n```json\n{paper_list}\n```"


if __name__ == '__main__':
    # 首先读取已处理的文件
    parser = argparse.ArgumentParser(description='step2 arguments')
    parser.add_argument('--model_name', type=str, default='Qwen3-Next-80B-A3B-Instruct')
    parser.add_argument('--engine', type=str, default='local_api', choices=['local_api', "api"])
    parser.add_argument('--journal_name', type=str, default="Advanced_Composites_and_Hybrid_Materials")
    # parser.add_argument('--md_path', type=str, default='solved_pdf/actamat')
    # parser.add_argument('--input_path', type=str, default='output_file/actamat/tetrahedron.json')
    # parser.add_argument('--output_path', type=str, default='output_file/actamat/images_function_deepseekv3.jsonl')
    args = parser.parse_args()

    md_path = f"solved_pdf/{args.journal_name}"
    input_path = f"output_file/{args.journal_name}/processed_data/tetrahedron.json"
    output_path = f"output_file/{args.journal_name}/raw_data/images_function_{args.model_name}.jsonl"
    # solved_path = f"output_file/{args.journal_name}/images_function.json"
    
    solved_list = []
    if os.path.exists(output_path):
        solved_list = [data['doi'] for data in [eval(line) for line in open(output_path, 'r', encoding='utf8')]]
    # if os.path.exists(solved_path):
    #     _solved_list = [doi for doi in json.load(open(solved_path, 'r', encoding='utf8'))]
    # solved_list.extend(_solved_list)
    f_o = open(output_path, "a+", encoding="utf-8")
    # 遍历ti6al4v_pdf文件夹下的所有文件

    with open(input_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    for doi, core_information in tqdm(json_dict.items()):
        if doi in solved_list:
            continue
        if core_information and not "logical chain" in core_information:
            continue
        if core_information and core_information["logical chain"]:
            paper_list = json.load(open(f"{md_path}/{doi}/auto/{doi}_content_list.json", "r", encoding="utf-8"))
            for idx, entry in enumerate(paper_list):
                entry.pop('page_idx', None)  # Remove 'page_idx' if it exists
                entry.pop('bbox', None)  # Remove 'section_idx' if it exists
                entry['block_index'] = idx  # Add index
            model_input = INPUT_TEMPLATE.format(instruction=instruction, paper_list=paper_list)
            # print(model_input)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": model_input}
            ]
            if args.engine == "api":
                output = generate_answer_api(messages, args.model_name)
            else:
                output = generate_answer_local_api(messages, args.model_name)
            
            if not output == None:
                content = {"doi": doi, "content": output}
                f_o.write(json.dumps(content, ensure_ascii=False) + "\n")
                f_o.flush()
            
        else:
            print(f"Skipping {doi} due to missing logical chain or empty core information.")
    f_o.close()