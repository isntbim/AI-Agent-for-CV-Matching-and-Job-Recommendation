"""Build the local V4 extractor for Ollama: merge the V4 QLoRA adapter into full-precision Qwen2.5-7B-Instruct on
the CPU, convert to a 16-bit GGUF with llama.cpp, quantise to Q4_K_M, then `ollama create` from the GGUF.

  HF_HOME=D:/hf-cache python -m scripts.qwen_build_ollama [--name qwen-v4] [--quant Q4_K_M]
  HF_HOME=D:/hf-cache python -m scripts.qwen_build_ollama --name qwen-o5 --adapter output/o5v3/runs/qlora_r16_o5v3/adapter

Ollama 0.40 cannot import Qwen2 safetensors (it routes them to the MLX engine), hence the llama.cpp step.
Needs llama.cpp's convert_hf_to_gguf.py (+ gguf-py, sentencepiece in this venv) and a llama-quantize binary.
The merged model is the same network the A100 evaluation ran (vLLM applied this adapter to this base);
only the quantisation differs, so re-score the test set after building (scripts/qwen_local_generate.py + qwen_eval).
"""

import argparse
import json
import shutil
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


def merge_lora(base: Path, adapter: Path, out: Path) -> None:
    """W += (alpha / r) * B @ A for every LoRA pair, one safetensors shard at a time (peak RAM about one shard, not the
    whole 15 GB model that PEFT's merge_and_unload holds). Plain LoRA only: no DoRA, rsLoRA, bias or extra modules."""
    import torch
    from safetensors.torch import load_file, save_file

    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert not (cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("modules_to_save")) and cfg.get("bias") == "none", cfg
    lora = load_file(adapter / "adapter_model.safetensors")
    scale, pairs = cfg["lora_alpha"] / cfg["r"], {}
    for k in lora:
        if ".lora_A." in k:  # base_model.model.<module>.lora_A.weight -> <module>.weight
            pairs[k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")] = (k, k.replace("lora_A", "lora_B"))
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy(f, out / f.name)  # config, tokenizer, shard index: the shard names stay the same
    done = 0
    for shard in sorted(base.glob("*.safetensors")):
        tensors = load_file(shard)
        for name, (a, b) in pairs.items():
            if name in tensors:
                w = tensors[name]
                tensors[name] = (w.float() + scale * (lora[b].float() @ lora[a].float())).to(w.dtype)
                done += 1
        save_file(tensors, out / shard.name, metadata={"format": "pt"})
        del tensors
    assert done == len(pairs), f"merged {done} of {len(pairs)} LoRA pairs"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, default=Path("D:/models"))
    p.add_argument("--name", default="qwen-v4")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--adapter", type=Path, default=ADAPTER, help="LoRA to merge (default: the V4 extractor)")
    p.add_argument("--llama-cpp", type=Path, default=Path("D:/tools/llama.cpp"))
    p.add_argument("--quantize-bin", type=Path, default=Path("D:/tools/llama-bin/llama-quantize.exe"))
    args = p.parse_args()
    merged = args.models / f"{args.name}-merged"
    f16 = args.models / f"{args.name}-f16.gguf"
    gguf = args.models / f"{args.name}-{args.quant.lower()}.gguf"
    if not (merged / "config.json").exists():
        from huggingface_hub import snapshot_download

        merge_lora(Path(snapshot_download(BASE)), args.adapter, merged)  # the eval built prompts with the base tokenizer
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
