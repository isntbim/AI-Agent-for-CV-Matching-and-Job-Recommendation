import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from scripts.qwen_build_ollama import merge_lora  # noqa: E402


def test_merge_lora_adds_scaled_low_rank_update(tmp_path):
    base, adapter, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "out"
    base.mkdir(), adapter.mkdir()
    w, other = torch.randn(4, 3).to(torch.bfloat16), torch.randn(5)
    safetensors.save_file({"model.layers.0.self_attn.q_proj.weight": w, "model.norm.weight": other},
                          base / "model-00001-of-00001.safetensors")
    (base / "config.json").write_text("{}", encoding="utf-8")
    a, b = torch.randn(2, 3), torch.randn(4, 2)
    safetensors.save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
                           "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b},
                          adapter / "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "bias": "none"}), encoding="utf-8")

    merge_lora(base, adapter, out)

    merged = safetensors.load_file(out / "model-00001-of-00001.safetensors")
    expected = (w.float() + 2 * (b @ a)).to(torch.bfloat16)  # alpha / r = 2
    assert torch.equal(merged["model.layers.0.self_attn.q_proj.weight"], expected)
    assert torch.equal(merged["model.norm.weight"], other)  # modules without a LoRA pair are copied unchanged
    assert (out / "config.json").exists()


def test_merge_lora_rejects_dora(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "bias": "none",
                                                                          "use_dora": True}), encoding="utf-8")
    with pytest.raises(AssertionError):
        merge_lora(tmp_path, tmp_path / "adapter", tmp_path / "out")
