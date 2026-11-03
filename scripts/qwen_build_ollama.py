"""Build the local V4 extractor for Ollama: merge the V4 QLoRA adapter into full-precision Qwen2.5-7B-Instruct on
the CPU, convert to a 16-bit GGUF with llama.cpp, quantise to Q4_K_M, then `ollama create` from the GGUF.

  HF_HOME=D:/hf-cache python -m scripts.qwen_build_ollama [--name qwen-v4] [--quant Q4_K_M]

Ollama 0.40 cannot import Qwen2 safetensors (it routes them to the MLX engine), hence the llama.cpp step.
Needs llama.cpp's convert_hf_to_gguf.py (+ gguf-py, sentencepiece in this venv) and a llama-quantize binary.
The merged model is the same network the A100 evaluation ran (vLLM applied this adapter to this base);
only the quantisation differs, so re-score the test set after building (scripts/qwen_local_generate.py + qwen_eval).
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = ROOT / "output/review2/qwen/v4/adapter"
MODELFILE = """FROM {gguf}
PARAMETER num_ctx 32768
PARAMETER temperature 0
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, default=Path("D:/models"))
    p.add_argument("--name", default="qwen-v4")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--llama-cpp", type=Path, default=Path("D:/tools/llama.cpp"))
    p.add_argument("--quantize-bin", type=Path, default=Path("D:/tools/llama-bin/llama-quantize.exe"))
    args = p.parse_args()
    merged = args.models / f"{args.name}-merged"
    f16 = args.models / f"{args.name}-f16.gguf"
    gguf = args.models / f"{args.name}-{args.quant.lower()}.gguf"
    if not (merged / "config.json").exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
        PeftModel.from_pretrained(base, ADAPTER).merge_and_unload().save_pretrained(
            merged, safe_serialization=True, max_shard_size="4GB")
        AutoTokenizer.from_pretrained(BASE).save_pretrained(merged)  # the eval built prompts with the base tokenizer
        print("merged ->", merged)
    if not gguf.exists():
        subprocess.run([sys.executable, str(args.llama_cpp / "convert_hf_to_gguf.py"), str(merged),
                        "--outtype", "f16", "--outfile", str(f16)], check=True)
        subprocess.run([str(args.quantize_bin), str(f16), str(gguf), args.quant], check=True)
        f16.unlink()  # 15 GB intermediate
    modelfile = args.models / f"Modelfile.{args.name}"
    modelfile.write_text(MODELFILE.format(gguf=gguf.as_posix()), encoding="utf-8")
    subprocess.run(["ollama", "create", args.name, "-f", str(modelfile)], check=True)


if __name__ == "__main__":
    main()
