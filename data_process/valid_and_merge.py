# 对mechanism数据进行合规性检测并将三个文件合并
import json
import os

# ---------------------------
#  Part 1: Mechanism Validator
# ---------------------------

def normalize_mechanism_entry(entry):
    """
    在字段检查前做字段名修正 + 数据结构归一化
    """

    # ------------------------------------------------------
    # 1) 修正 images 字段的 image_description -> image description
    # ------------------------------------------------------
    images = entry.get("images")
    if isinstance(images, list):
        for img in images:
            if not isinstance(img, dict):
                continue
            if "image_description" in img:
                img["image description"] = img["image_description"]
                # 删掉旧字段
                del img["image_description"]
    elif isinstance(images, dict):
        # 兼容错误格式：images: {...}
        if "image_description" in images:
            images["image description"] = images["image_description"]
            del images["image_description"]
        # 改成 list
        entry["images"] = [images]

    # ------------------------------------------------------
    # 2) 统一 experiment 格式：dict → list
    # ------------------------------------------------------
    # exp = entry.get("experiment")
    # if isinstance(exp, dict):
    #     entry["experiment"] = [exp]

    # elif isinstance(exp, list):
    #     # 有时 list 中会有非 dict，提前修一下
    #     entry["experiment"] = [e for e in exp if isinstance(e, dict)]

    # else:
    #     # 若格式完全不对，直接置空防止报错
    #     entry["experiment"] = []


    return entry



def check_field_completeness(obj, required_fields, path):
    errors = []
    if not isinstance(obj, dict):
        errors.append(f"{path} is not a dict")
    else:
        missing = [f for f in required_fields if f not in obj]
        if missing:
            errors.append(f"{path} missing fields: {missing}")
    return errors


def check_list_of_dicts(obj, required_fields, path):
    errors = []
    if not isinstance(obj, list):
        errors.append(f"{path} is not a list")
    else:
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                errors.append(f"{path}[{i}] is not a dict")
                continue
            missing = [f for f in required_fields if f not in item]
            if missing:
                errors.append(f"{path}[{i}] missing fields: {missing}")
    return errors


def check_mechanism_entry(entry, doi):
    errors = []
    entry = normalize_mechanism_entry(entry)
    # Experiment
    if entry.get("experiment"):
        errors += check_field_completeness(
            entry["experiment"],
            ["name", "type", "parameters", "result", "source"],
            f"[DOI {doi}] experiment"
        )

    # Images
    if entry.get("images"):
        errors += check_list_of_dicts(
            entry["images"],
            ["block_index", "image description", "source"],
            f"[DOI {doi}] images"
        )

    # Referenced knowledge
    if entry.get("referenced_knowledge"):
        errors += check_list_of_dicts(
            entry["referenced_knowledge"],
            ["content", "reference", "source"],
            f"[DOI {doi}] referenced_knowledge"
        )

    # Domain knowledge
    if entry.get("domain_knowledge"):
        errors += check_list_of_dicts(
            entry["domain_knowledge"],
            ["content", "source"],
            f"[DOI {doi}] domain_knowledge"
        )

    # Mechanism
    mech = entry.get("mechanism")
    if mech:
        errors += check_field_completeness(
            mech,
            ["description", "source", "reasoning_chain"],
            f"[DOI {doi}] mechanism"
        )

        if isinstance(mech, dict) and isinstance(mech.get("reasoning_chain"), list):
            for i, rc in enumerate(mech["reasoning_chain"]):
                if not isinstance(rc, dict):
                    errors.append(f"[DOI {doi}] reasoning_chain[{i}] is not a dict")
                    continue
                missing = [f for f in ["type", "statement"] if f not in rc]
                if missing:
                    errors.append(
                        f"[DOI {doi}] reasoning_chain[{i}] missing fields: {missing}"
                    )
        else:
            errors.append(f"[DOI {doi}] mechanism reasoning_chain is not a list")
    else:
        errors.append(f"[DOI {doi}] mechanism is missing or empty")

    return errors


def validate_and_filter_mechanism_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_errors = []
    error_dois = set()
    valid_data = {}

    for doi, mechanisms in data.items():

        # ensure list
        if isinstance(mechanisms, dict):
            mechanisms = [mechanisms]

        mechanism_errors = []
        for mechanism in mechanisms:
            if not isinstance(mechanism, dict):
                mechanism_errors.append(f"[DOI {doi}] Mechanism entry is not a dict")
                break
            mechanism_errors.extend(check_mechanism_entry(mechanism, doi))

        if mechanism_errors:
            print(f"❌ Errors found in DOI {doi}:")
            for err in mechanism_errors:
                print(f"   - {err}")
            all_errors.extend(mechanism_errors)
            error_dois.add(doi)
        else:
            valid_data[doi] = mechanisms

    print(f"Total entries checked: {len(data)}")
    print(f"❌ Total entries with errors: {len(error_dois)}")
    print(f"✅ Total valid entries: {len(valid_data)}")

    return valid_data  # filtered mechanism


# ---------------------------
#  Part 2: Merge Three JSONs
# ---------------------------

def merge_three_jsons(journal_name, tetra_path, imgfunc_path, mech_path_filtered):
    # Load JSONs
    tetrahedrons = json.load(open(tetra_path, "r", encoding='utf8'))
    images_functions = json.load(open(imgfunc_path, "r", encoding='utf8'))

    # Combine filtered mechanism data
    mechanisms = mech_path_filtered

    # Output folder
    output_dir = f"final_output/{journal_name}"
    os.makedirs(output_dir, exist_ok=True)

    for doi, mechanism in mechanisms.items():
        if not mechanism:
            print(f"Mechanism for {doi} is empty, skipping...")
            continue

        tetra = tetrahedrons.get(doi, {})
        imgfunc = images_functions.get(doi, {})

        if not tetra or not imgfunc:
            print(f"Incomplete data for {doi}, skipping...")
            continue

        knowledge = {
            "tetrahedron": tetra,
            "images_function": imgfunc,
            "mechanism": mechanism
        }

        output_path = os.path.join(output_dir, doi + ".json")
        with open(output_path, "w", encoding='utf8') as f:
            json.dump(knowledge, f, indent=4, ensure_ascii=False)


# ---------------------------
#  Part 3: Full Pipeline
# ---------------------------

def run_full_pipeline(journal_name):
    base = f"output_file/{journal_name}"

    # Step 1: validate mechanism.json
    mechanism_filtered = validate_and_filter_mechanism_json(f"{base}/processed_data/mechanism.json")

    # Step 2: merge three files
    merge_three_jsons(
        journal_name,
        tetra_path=f"{base}/processed_data/tetrahedron.json",
        imgfunc_path=f"{base}/processed_data/images_function.json",
        mech_path_filtered=mechanism_filtered
    )

    print("🎉 All tasks completed!")


# 执行整个流程
run_full_pipeline("Advanced_Energy_Materials")

