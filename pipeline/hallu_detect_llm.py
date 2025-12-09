import os
import json
import re
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from component.utils import generate_answer_api, generate_answer_local_api

#############################################
# 配置
#############################################


instruction = open("instruction/hallu_detect.md", "r", encoding="utf-8").read()
system_message = "You are an expert in the field of materials science."
restrict = "\n\nPlease just output the json format as the example above. Do not output any other information."

#############################################
# 工具函数：根据 source 列表生成先验文本
#############################################

def load_paper_content(doi, journal):
    """
    读取 solved_pdf/{journal}/{doi}/auto/{doi}_content_list.json
    返回 text 列表（按出现顺序）
    """
    json_file = f"solved_pdf/{journal}/{doi}/auto/{doi}_content_list.json"
    if not os.path.exists(json_file):
        return []
    paper_list = json.load(open(json_file, "r", encoding="utf-8"))

    for idx, entry in enumerate(paper_list):
        entry.pop('page_idx', None)  # Remove 'page_idx' if it exists
        entry.pop('bbox', None)  # Remove 'section_idx' if it exists
        entry['block_index'] = idx  # Add index

    return paper_list

def extract_json(content):
    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError:
            print("content", content)
            json_obj = None
    else:
        try:
            json_obj = json.loads(content)
        except json.JSONDecodeError:
            print("content", content)
            json_obj = None  # 表示 None 或解析失败
    return json_obj

def fetch_source_text(all_texts, source_indices):
    """
    source_indices 是一组数字，比如 [13,29]
    则返回 all_texts 中对应 index 的文本拼接
    如果 index 超出范围，跳过
    """
    collected = []
    for idx in source_indices:
        if 0 <= idx < len(all_texts):
            try:
                collected.append(all_texts[idx])
            except:
                continue
    return collected

def generate_hallu(engine, model_name, prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    if engine == "api":
        output = generate_answer_api(messages, model_name)
    else:
        output = generate_answer_local_api(messages, model_name)

    hallu = extract_json(output)

    return hallu


def judge_and_write(hypothesis, source_ids, container, all_texts, engine, model_name):
    """
    通用判断函数：
    - hypothesis: 待判断文本
    - source_ids: 文本出处编号列表
    - container: 往哪个 dict 写入 hallu_detection 的那个 dict
    """

    premise = fetch_source_text(all_texts, source_ids)
    
    prompt = instruction.format(premise=premise, hypothesis=hypothesis) + restrict
    hallu_json = generate_hallu(engine, model_name, prompt)
    container['hallu_detection'] = hallu_json


def process_mechanism_with_llm(mech_json, all_texts, engine, model_name):
    """
    遍历 mech_json，针对需要检测的字段调用 judge_and_write()
    """
    for block in mech_json:

        # 1. experiment
        exp = block.get("experiment")
        if exp and "result" in exp and "source" in exp:
            consistency = exp.get('hallu_detection', {}).get('consistency')
            if not consistency:
                judge_and_write(exp["result"], exp["source"], exp, all_texts, engine, model_name)

        # 2. images
        imgs = block.get("images", [])
        if imgs:
            for img in imgs:
                if "image description" in img and "source" in img:
                    consistency = img.get('hallu_detection', {}).get('consistency')
                    if not consistency:
                        judge_and_write(img["image description"], img["source"], img, 
                                    all_texts, engine, model_name)

        # 3. referenced_knowledge
        refks = block.get("referenced_knowledge", [])
        if refks:
            for refk in refks:
                consistency = refk.get('hallu_detection', {}).get('consistency')
                if not consistency:
                    judge_and_write(refk.get("content"), refk.get("source"), refk,
                                all_texts, engine, model_name)

        # 4. domain_knowledge
        dks = block.get("domain_knowledge", [])
        if dks:
            for dk in dks:
                consistency = dk.get('hallu_detection', {}).get('consistency')
                if not consistency:
                    judge_and_write(dk.get("content"), dk.get("source"), dk,
                                all_texts, engine, model_name)

        # 5. mechanism.description
        mech = block.get("mechanism")
        if mech and "description" in mech and "source" in mech:
            consistency = mech.get('hallu_detection', {}).get('consistency')
            if not consistency:
                judge_and_write(mech["description"], mech["source"], mech, 
                            all_texts, engine, model_name)

    return mech_json


def main():
    parser = argparse.ArgumentParser(description='hallucination detection arguments')
    parser.add_argument('--model_name', type=str, default='Qwen3-Next-80B-A3B-Instruct', choices=['Qwen3-Next-80B-A3B-Instruct', 'qwen-plus-latest'])
    parser.add_argument('--engine', type=str, default='local_api', choices=['local_api', "api"])
    parser.add_argument('--journal_name', type=str, default='Advanced_Materials')
    args = parser.parse_args()

    journal = args.journal_name
    final_dir = f"output_file/{journal}/hallucination_slm"
    out_dir = f"output_file/{journal}/hallucination_llm"
    os.makedirs(out_dir, exist_ok=True)

    # 从out_dir获取已处理的doi列表，跳过已处理的
    processed_dois = set()
    for fname in os.listdir(out_dir):
        if fname.endswith(".json"):
            processed_dois.add(fname.replace(".json", ""))

    for fname in tqdm(os.listdir(final_dir)):
        if not fname.endswith(".json"):
            continue

        doi = fname.replace(".json", "")
        if doi in processed_dois:
            continue

        mech_file = os.path.join(final_dir, fname)

        konwledge_json = json.load(open(mech_file, "r", encoding="utf-8"))
        mech_json = konwledge_json.get('mechanism', [])
        all_texts = load_paper_content(doi, journal)

        if not mech_json:
            continue

        # 逐 pair 调用模型并写回 mech_json
        new_json = {}
        updated_mech = process_mechanism_with_llm(mech_json, all_texts, args.engine, args.model_name)
        new_json['mechanism'] = updated_mech

        # 将每篇文献单独保存为 json 文件
        out_path = os.path.join(out_dir, f"{doi}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_json, f, indent=2, ensure_ascii=False)

        # print(f"Processed and saved hallucination results for {doi} -> {out_path}")


if __name__ == "__main__":
    main()
