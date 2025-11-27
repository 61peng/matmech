import os
import json
import argparse
from transformers import pipeline, AutoTokenizer
from component.utils import tqdm

#############################################
# 参数解析
#############################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=str, required=True,
                        help="Journal name, e.g. Advanced_Materials")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU id, e.g. 0 or 7")
    parser.add_argument("--model", type=str, default="model/hallucination_evaluation_model",
                        help="Path to hallucination evaluation model")
    return parser.parse_args()


#############################################
# 加载论文文本
#############################################

def load_paper_content(doi, journal):
    json_file = f"solved_pdf/{journal}/{doi}/auto/{doi}_content_list.json"
    if not os.path.exists(json_file):
        return []
    paper_list = json.load(open(json_file, "r", encoding="utf-8"))
    all_texts = []
    for x in paper_list:
        t = x.get("type")
        if t in ["text", "equation"]:
            if "text" in x:
                all_texts.append(x["text"])
        elif t == "image":
            all_texts.append(str(x.get("img_caption")))
        elif t == "table":
            all_texts.append(str(x.get("table_caption")))
    return all_texts


def fetch_source_text(all_texts, source_indices):
    collected = []
    for idx in source_indices:
        if 0 <= idx < len(all_texts):
            collected.append(all_texts[idx])
    return "\n".join(collected) if collected else ""


#############################################
# Pair 收集工具
#############################################

def add_pair(pairs, container, field_name, premise, hypothesis):
    pairs.append({
        "premise": premise,
        "hypothesis": hypothesis,
        "container": container,
        "field_name": field_name
    })


def gather_pairs(mech_json, all_texts):
    pairs = []
    for block in mech_json:

        # experiment
        exp = block.get("experiment")
        if isinstance(exp, dict):  # 单个 dict => list（容错）
            exp = [exp]
        if isinstance(exp, list):
            for e in exp:
                if "result" in e and "source" in e:
                    prem = fetch_source_text(all_texts, e["source"])
                    hyp = e["result"]
                    if prem and hyp:
                        add_pair(pairs, e, "result", prem, hyp)

        # images
        imgs = block.get("images", [])
        if isinstance(imgs, dict):
            imgs = [imgs]
        for img in imgs:
            if "image description" in img and "source" in img:
                prem = fetch_source_text(all_texts, img["source"])
                hyp = img["image description"]
                if prem and hyp:
                    add_pair(pairs, img, "image description", prem, hyp)

        # referenced_knowledge
        for refk in block.get("referenced_knowledge", []):
            prem = fetch_source_text(all_texts, refk.get("source", []))
            hyp = refk.get("content")
            if prem and hyp:
                add_pair(pairs, refk, "content", prem, hyp)

        # domain_knowledge
        for dk in block.get("domain_knowledge", []):
            prem = fetch_source_text(all_texts, dk.get("source", []))
            hyp = dk.get("content")
            if prem and hyp:
                add_pair(pairs, dk, "content", prem, hyp)

        # mechanism.description
        mech = block.get("mechanism")
        if mech and "description" in mech and "source" in mech:
            prem = fetch_source_text(all_texts, mech["source"])
            hyp = mech["description"]
            if prem and hyp:
                add_pair(pairs, mech, "description", prem, hyp)

    return pairs


#############################################
# 主流程
#############################################

def run(journal, gpu, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    INPUT_DIR = f"final_output/{journal}"
    OUTPUT_DIR = f"output_file/{journal}/hallucination_slm"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PROMPT_TEMPLATE = (
        "<pad> Determine if the hypothesis is true given the premise?\n\n"
        "Premise: {premise}\n\nHypothesis: {hypothesis}"
    )

    # classifier pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=AutoTokenizer.from_pretrained("model/flan-t5-base"),
        trust_remote_code=True
    )

    # 遍历每篇 paper
    for fname in tqdm(os.listdir(INPUT_DIR)):
        if not fname.endswith(".json"):
            continue

        doi = fname[:-5]
        mech_file = os.path.join(INPUT_DIR, fname)
        knowledge_json = json.load(open(mech_file, "r", encoding="utf-8"))
        mech_json = knowledge_json.get("mechanism", [])

        # 加载论文全文
        all_texts = load_paper_content(doi, journal)

        # 收集所有 pair
        pairs = gather_pairs(mech_json, all_texts)
        if not pairs:
            continue

        # 构建 prompts
        prompts = [
            PROMPT_TEMPLATE.format(premise=p["premise"], hypothesis=p["hypothesis"])
            for p in pairs
        ]

        batch_scores = classifier(prompts, top_k=None)

        # 写回检测结果
        for pair, score_set in zip(pairs, batch_scores):
            consistent_score = next(
                (s["score"] for s in score_set if s["label"] == "consistent"),
                0.0
            )
            consistency = consistent_score > 0.5

            container = pair["container"]
            field = pair["field_name"]
            container["hallu_detection"] = {
                "consistency": consistency,
                "confidence": float(consistent_score)
            }
            container["source_text"] = pair["premise"]

        # 保存
        out_path = os.path.join(OUTPUT_DIR, f"{doi}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(knowledge_json, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    run(args.journal, args.gpu, args.model)


if __name__ == "__main__":
    main()
