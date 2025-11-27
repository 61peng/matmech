import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from component.utils import generate_answer_api, generate_answer_local_api
import argparse
from pathlib import Path
from tqdm import tqdm


system_message = 'You are an expert in the field of materials science, please follow the following steps to output the core knowledge of the paper.'

RE_list = ["TEM", "EDX", "SEM",  "XPS", "AFM", "STM", "Raman", "UV-Vis", "Microstructure", "Micrographs", r"\upmu", "Micrograph",
           "XRD"]

instruction = open("instruction/tetrahedron.md", "r", encoding="utf-8").read()

restrict = "\n\nPlease just output the json format as the example above or None. Do not output any other information."

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='step1 arguments')
    parser.add_argument('--model_name', type=str, default='Qwen3-Next-80B-A3B-Instruct', choices=['qwen-plus-latest', 'Qwen3-Next-80B-A3B-Instruct'])
    parser.add_argument('--engine', type=str, default='local_api', choices=['local_api', "api"])
    parser.add_argument('--journal_name', type=str, default="Advanced_Composites_and_Hybrid_Materials")
    # parser.add_argument('--output_path', type=str, default='output_file/nama/raw_tetrahedron_deepseekv3.jsonl')
    args = parser.parse_args()

    input_path = f"solved_pdf/{args.journal_name}"
    output_path = f"output_file/{args.journal_name}/raw_data/tetrahedron_{args.model_name}.jsonl"
    paper_type_path = f"output_file/{args.journal_name}/processed_data/paper_type.json"
    # solved_path = f"output_file/{args.journal_name}/tetrahedron.json"

    try:
        with open(paper_type_path, 'r', encoding='utf-8') as f:
            paper_type_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: paper_type.json not found at {paper_type_path}")
        exit()

    # 提取 classification 为 "FourElementRelationshipStudy" 的论文 DOI
    target_dois = []
    for paper in paper_type_data:
        # 假设 "content" 是一个可以被 eval 的字符串
        try:
            content_dict = paper.get('content', '{}')
            if content_dict.get("classification") == "FourElementRelationshipStudy":
                target_dois.append(paper.get('doi'))
        except:
            continue # 忽略无法解析的条目

    print(f"Found {len(target_dois)} papers with classification 'FourElementRelationshipStudy'.")

    # 已处理的 DOI 列表
    solved_list = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            solved_list = [json.loads(line)['doi'] for line in f]
    # if os.path.exists(solved_path):
    #     solved_list = [doi for doi in json.load(open(solved_path, 'r', encoding='utf8'))]
    # 加载模型
    print(f"已处理{len(solved_list)}篇论文")


    f_o = open(output_path, "a+", encoding="utf-8")

    # 收集所有待处理的文件
    # 遍历筛选出的 DOI
    for doi in tqdm(target_dois):
        if doi in solved_list:
            continue
        local_md_dir = Path(os.path.join(input_path, doi, 'auto'))
        if any(local_md_dir.glob("*.md")):
            if doi not in solved_list:
                filepath = os.path.join(local_md_dir, f"{doi}.md")

                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                model_input = instruction + content + restrict
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": model_input}
                ]

                if args.engine == "api":
                    output = generate_answer_api(messages, args.model_name, max_tokens=2048)
                elif args.engine == "local_api":
                    output = generate_answer_local_api(messages, args.model_name, max_tokens=2048)


                if output is not None:
                    f_o.write(json.dumps({"doi": doi, "content": output}, ensure_ascii=False) + "\n")
                    f_o.flush()
        
        else:
            print(f"File not found for DOI: {doi}")

    f_o.close()
    print("Processing completed.")