import os

import json
import argparse
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



def extract_sentences_from_mech_json(mech_list, doi, journal):
    """
    Extract sentences from the mechanism list.
    Fields kept: sentence, doi, journal, type.
    """
    results = []

    for block in mech_list:
        # referenced_knowledge
        refks = block.get("referenced_knowledge", [])
        if refks:
            for refk in refks:
                if "content" in refk:
                    results.append({
                        "doi": doi,
                        "journal": journal,
                        "type": "referenced_knowledge",
                        "sentence": refk["content"]
                    })

        # domain_knowledge
        dks = block.get("domain_knowledge", [])
        if dks:
            for dk in dks:
                if "content" in dk:
                    results.append({
                        "doi": doi,
                        "journal": journal,
                        "type": "domain_knowledge",
                        "sentence": dk["content"]
                    })

        # mechanism.description
        mech = block.get("mechanism")
        if mech and "description" in mech:
            results.append({
                "doi": doi,
                "journal": journal,
                "type": "mechanism_description",
                "sentence": mech["description"]
            })

    return results


def extract_all_sentences_from_all_journals(journal_list, base_path="final_output"):
    """
    Extract all sentences across all journals.
    """
    all_entries = []
    sentence_id = 0

    for journal in journal_list:
        journal_dir = os.path.join(base_path, journal)
        if not os.path.exists(journal_dir):
            print(f"[WARN] Directory not found: {journal_dir}")
            continue

        print(f"\n>>> Loading journal: {journal}")

        for fname in tqdm(os.listdir(journal_dir), desc=f"Reading {journal}"):
            if not fname.endswith(".json"):
                continue

            doi = fname[:-5]
            fpath = os.path.join(journal_dir, fname)

            try:
                data = json.load(open(fpath, "r", encoding="utf-8"))
            except:
                print(f"[WARN] Failed to load: {fpath}")
                continue

            mech_list = data.get("mechanism")
            if not isinstance(mech_list, list):
                continue

            extracted = extract_sentences_from_mech_json(mech_list, doi, journal)
            for s in extracted:
                s["sentence_id"] = sentence_id
                all_entries.append(s)
                sentence_id += 1

    print(f"\nTotal sentences extracted across journals: {len(all_entries)}")
    return all_entries




def compute_embeddings(entries, batch_size=32):
    """
    Compute embeddings using sentence-transformers all-mpnet-base-v2.
    """
    model = SentenceTransformer("model/all-mpnet-base-v2")
    sentences = [e["sentence"] for e in entries]

    all_embeds = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
        batch = sentences[i:i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True)
        all_embeds.append(emb)

    all_embeds = np.vstack(all_embeds).astype("float32")
    print(f"Embedding shape: {all_embeds.shape}")
    return all_embeds


def save_embeddings(path, embeddings):
    np.save(path, embeddings)
    print(f"Saved embeddings to {path}")


def load_embeddings(path):
    print(f"Loading embeddings from {path}")
    return np.load(path)


def build_faiss_index_cpu(embeddings):
    """
    Build CPU flat IP index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)

    ids = np.arange(embeddings.shape[0]).astype("int64")
    print("Adding vectors to FAISS CPU index...")
    index.add_with_ids(embeddings, ids)
    print("Index size:", index.ntotal)
    return index


def build_faiss_index_gpu(embeddings):
    """
    Build GPU flat IP index (cosine similarity).
    """
    dim = embeddings.shape[1]

    # Try create GPU resources
    try:
        res = faiss.StandardGpuResources()
    except Exception as e:
        raise RuntimeError("FAISS GPU not installed or CUDA mismatch") from e

    cpu_index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    index = faiss.IndexIDMap(gpu_index)

    ids = np.arange(embeddings.shape[0]).astype("int64")
    print("Adding vectors to FAISS GPU index...")
    index.add_with_ids(embeddings, ids)

    print("Index size:", index.ntotal)
    return index


def save_faiss_index(index, path):
    """
    Save FAISS index (GPU or CPU).
    Converts GPU index to CPU automatically.
    """
    # try:
    #     cpu_index = faiss.index_gpu_to_cpu(index)
    # except Exception:
    cpu_index = index

    faiss.write_index(cpu_index, path)
    print(f"FAISS index saved to {path}")


def load_faiss_index(path):
    """
    Load FAISS CPU index, then try move to GPU if available.
    """
    print(f"Loading FAISS index from {path}")
    cpu_index = faiss.read_index(path)

    # Try GPU
    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("Loaded FAISS GPU index.")
        return gpu_index
    except Exception as e:
        print(f"[WARN] GPU FAISS not available, using CPU index. Reason: {e}")
        return cpu_index



def retrieve_for_journal(journal, entries, sims, nn_ids, top_k, output_base="output_file"):
    """
    Save retrieval results into output_file/{journal}/processed_data/{doi}.json
    """
    print(f"\n>>> Saving retrieval results for journal: {journal}")

    out_dir = os.path.join(output_base, journal, "processed_data")
    os.makedirs(out_dir, exist_ok=True)

    # Find all entries belonging to this journal
    idxs = [i for i, e in enumerate(entries) if e["journal"] == journal]

    results_by_doi = {}

    for i in idxs:
        src = entries[i]
        src_doi = src["doi"]

        candidates = []
        count = 0

        # Iterate the NN candidate list
        for cid, sim in zip(nn_ids[i], sims[i]):
            if cid < 0 or cid == i:
                continue

            tgt = entries[cid]

            # Exclude same DOI (same paper)
            if tgt["doi"] == src_doi:
                continue

            candidates.append({
                "sentence_id": tgt["sentence_id"],
                "doi": tgt["doi"],
                "sentence": tgt["sentence"],
                "similarity": float(sim)
            })

            count += 1
            if count >= top_k:
                break

        record = {
            "sentence_id": src["sentence_id"],
            "doi": src_doi,
            "type": src["type"],
            "sentence": src["sentence"],
            "candidates": candidates
        }

        results_by_doi.setdefault(src_doi, []).append(record)

    # Save per DOI
    for doi, recs in results_by_doi.items():
        out_path = os.path.join(out_dir, f"{doi}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)

    print(f"Saved in: {out_dir}")


def faiss_batch_search(index, embeddings, k, batch_size=10000):
    """
    Perform FAISS search in batches with progress bar.
    Suitable for large datasets (e.g., 600k vectors).
    """
    n = embeddings.shape[0]

    all_sims = []
    all_ids = []

    for start in tqdm(range(0, n, batch_size), desc="FAISS batch search"):
        end = min(start + batch_size, n)
        batch = embeddings[start:end]

        sims, ids = index.search(batch, k)
        all_sims.append(sims)
        all_ids.append(ids)

    # Stack back
    all_sims = np.vstack(all_sims)
    all_ids = np.vstack(all_ids)

    return all_sims, all_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # Journals to process
    journal_list = [
        "Advanced_Materials",
        "Advanced_Functional_Materials",
        "Nature_Materials",
        "Nano_Letters",
        "Acta_Materialia",
        "Advanced_Composites_and_Hybrid_Materials",
        "Bioactive_Materials",
        "Biomaterials",
        "Journal_of_Advanced_Ceramics",
        "Journal_of_Materials_Science_&_Technology",
        "Materials_Characterization",
        "Progress_in_Organic_Coatings",
        "Rare_Metals",
        "Advanced_Energy_Materials",
        "Journal_of_Magnesium_and_Alloys"
    ]

    # Ensure cache exists
    os.makedirs("cache", exist_ok=True)

    # Step 1: extract sentences across all journals
    entries = extract_all_sentences_from_all_journals(journal_list)

    # Step 2: load or compute embeddings
    embed_path = "cache/embeddings.npy"
    if os.path.exists(embed_path):
        embeddings = load_embeddings(embed_path)
    else:
        embeddings = compute_embeddings(entries)
        save_embeddings(embed_path, embeddings)

    # Step 3: load or build index
    index_path = "cache/faiss.index"
    if os.path.exists(index_path):
        index = load_faiss_index(index_path)
    else:
        # try:
        #     index = build_faiss_index_gpu(embeddings)
        # except Exception as e:
        #     print("GPU index failed. Falling back to CPU.", e)
        index = build_faiss_index_cpu(embeddings)

        save_faiss_index(index, index_path)

    # Step 4: batch KNN search
    print("\nRunning batch KNN search...")
    # sims, nn_ids = index.search(embeddings, args.top_k + 10)
    sims, nn_ids = faiss_batch_search(index, embeddings, args.top_k + 5, batch_size=10000)


    # Step 5: save results by journal
    for journal in journal_list:
        retrieve_for_journal(journal, entries, sims, nn_ids, args.top_k)

    print("\nAll journals processed.")


if __name__ == "__main__":
    main()
