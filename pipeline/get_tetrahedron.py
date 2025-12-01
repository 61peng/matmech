import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from component.utils import generate_answer_api, generate_answer_local_api
import argparse
from pathlib import Path
from tqdm import tqdm


system_message = 'You are an expert in the field of materials science, please follow the following steps to output the core knowledge of the paper.'

instruction = open("instruction/tetrahedron.md", "r", encoding="utf-8").read()

restrict = "\n\nPlease just output the json format as the example above or None. Do not output any other information."

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='step1 arguments')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--engine', type=str, default='local_api', choices=['local_api', "api"])
    parser.add_argument('--journal_name', type=str)
    args = parser.parse_args()

    input_path = f"solved_pdf/{args.journal_name}"
    output_path = f"output_file/{args.journal_name}/raw_data/tetrahedron_{args.model_name}.jsonl"
    paper_type_path = f"output_file/{args.journal_name}/processed_data/paper_type.json"

    try:
        with open(paper_type_path, 'r', encoding='utf-8') as f:
            paper_type_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: paper_type.json not found at {paper_type_path}")
        exit()

    target_dois = []
    for paper in paper_type_data:
        # Suppose "content" is a dictionary stored as a string
        try:
            content_dict = paper.get('content', '{}')
            if content_dict.get("classification") == "FourElementRelationshipStudy":
                target_dois.append(paper.get('doi'))
        except:
            continue

    print(f"Found {len(target_dois)} papers has tetrahedron.")

    # Processed DOI List
    solved_list = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            solved_list = [json.loads(line)['doi'] for line in f]

    print(f"solved {len(solved_list)} papers")

    f_o = open(output_path, "a+", encoding="utf-8")

    # collect all files to be processed
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