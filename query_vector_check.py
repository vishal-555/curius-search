"""
Check that a machine (for example the Oracle VM) produces the same query vectors as another one (the laptop).

    python query_vector_check.py dump vm_vectors.npy        # run on each machine
    python query_vector_check.py compare laptop_vectors.npy vm_vectors.npy    # run anywhere, with both files

Cosine between the two machines should be about 1.0000 (0.9999 or higher). Lower means the model
loaded in a different precision or a different model version, and search results will drift.
"""

import sys

import numpy as np

QUERIES = [
    "using AI tools to become more ambitious",
    "how semiconductor companies compete",
    "advice for doing good research as a beginner",
    "AI safety and alignment research",
    "how to write clearly",
]


def dump(path):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    try:
        m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", model_kwargs={"dtype": torch.float32})
    except TypeError:
        m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", model_kwargs={"torch_dtype": torch.float32})
    m.float()
    print("torch", torch.__version__, "| model dtype", next(m.parameters()).dtype)
    v = m.encode(QUERIES, prompt_name="query", normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(path, v)
    print("first 5 values of the first query:", np.round(v[0][:5], 6))
    print(f"saved {v.shape} to {path}")


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    cos = (a * b).sum(axis=1)
    for q, c in zip(QUERIES, cos):
        print(f"  {c:.6f}  {q}")
    print(f"minimum cosine {cos.min():.6f}:", "OK, same vectors" if cos.min() > 0.9999 else "MISMATCH, check dtype and model version")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
