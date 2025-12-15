import os
import json
import re
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from component.utils import generate_answer_api, generate_answer_local_api



system_message = "You are an expert in materials science, skilled at detecting semantic and factual conflicts between scientific statements."

instruction_template = open("instruction/conflict_detect.md", "r", encoding="utf-8").read()

restrict = "\n\nPlease return ONLY valid JSON output in the required format. No explanations outside JSON."


def extract_json(content):
    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def build_prompt(source_sentence, candidate_dicts):

    candidate_text = "\n".join([
        f"{c['cid']}: {c['sentence']}" for c in candidate_dicts
    ])

    prompt = instruction_template.format(
        source_sentence=source_sentence,
        candidate_list=candidate_text
    ) + restrict

    return prompt


def generate_conflict(engine, model_name, prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    if engine == "api":
        output = generate_answer_api(messages, model_name)
    else:
        output = generate_answer_local_api(messages, model_name)

    if output:
        output = extract_json(output)

    return output



def process_single_file(json_file, out_dir, sim_thres, engine, model_name):
    """
    process conflict detection for each json in topk_retrieval
    """

    data = json.load(open(json_file, "r", encoding="utf-8"))

    results = []

    for item in data:
        source_sentence = item["sentence"]

        # filter candidates whose similarity > threshold
        cand_filtered = []
        for c in item.get("candidates", []):
            if c["similarity"] >= sim_thres:
                cand_filtered.append({
                    "cid": c["sentence_id"],
                    "sentence": c["sentence"],
                    "similarity": c["similarity"],
                    "doi": c["doi"],
                    "sentence_id": c["sentence_id"]
                })

        if len(cand_filtered) == 0:
            continue

        # construct prompt
        prompt = build_prompt(source_sentence, cand_filtered)

        conflict_json = generate_conflict(engine, model_name, prompt)
        
        if conflict_json:

            results.append({
                "source_sentence": source_sentence,
                "source_sentence_id": item["sentence_id"],
                "candidates": cand_filtered,
                "conflict_detection": conflict_json
            })

    # 保存输出
    doi = os.path.basename(json_file).replace(".json", "")
    out_path = os.path.join(out_dir, f"{doi}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Conflict detection pipeline")
    parser.add_argument("--journal_name", type=str, default="Nano_Letters")
    parser.add_argument("--sim_thres", type=float, default=0.90)
    parser.add_argument("--engine", type=str, default="api", choices=["api", "local_api"])
    parser.add_argument("--model_name", type=str, default="qwen-plus-latest")
    args = parser.parse_args()

    journal = args.journal_name

    input_dir = f"output_file/{journal}/topk_retrieval"
    out_dir = f"output_file/{journal}/conflict"
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    print(f"Found {len(files)} files in {input_dir}")

    solved_file = os.listdir(out_dir)

    for fname in tqdm(files, desc="Processing conflict detection"):
        if fname in solved_file:
            continue
        fpath = os.path.join(input_dir, fname)
        process_single_file(
            fpath, out_dir, args.sim_thres, args.engine, args.model_name
        )
        # print(f"Saved conflict result -> {out_path}")


if __name__ == "__main__":
    main()
