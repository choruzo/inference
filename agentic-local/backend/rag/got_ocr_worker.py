"""Standalone GOT-OCR2_0 inference worker.

Runs as a subprocess, never imported by the main FastAPI app, so torch/transformers
are only loaded into memory when a document actually needs this OCR path, and the
process can be started/stopped independently of the llama.cpp chat/embeddings models
sharing the same GPU (see GotOcrVramSwap in backend/rag/documents.py).

Usage: python got_ocr_worker.py <image_path> <model_dir> [ocr_type]
Prints recognized text to stdout; nothing else goes to stdout.
"""
from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: got_ocr_worker.py <image_path> <model_dir> [ocr_type]", file=sys.stderr)
        return 2
    image_path, model_dir = argv[0], argv[1]
    ocr_type = argv[2] if len(argv) > 2 else "ocr"

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        print(f"GOT-OCR2_0 requires torch/transformers: {exc}. Install backend/requirements-ocr-got.txt", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The checkpoint is published in bf16 (~1.44 GB); without an explicit dtype,
    # from_pretrained upcasts to fp32, which alone can exceed a 4 GB card once the
    # vision encoder's activations are added on top. bf16 matches the stored weights
    # and keeps this runnable on constrained VRAM (e.g. the GTX 1050 this targets).
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device,
        use_safetensors=True,
        torch_dtype=dtype,
        pad_token_id=tokenizer.eos_token_id,
    )
    model = model.eval().to(device=device, dtype=dtype)

    result = model.chat(tokenizer, image_path, ocr_type=ocr_type)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
