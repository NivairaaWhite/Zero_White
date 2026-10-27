"""export.py - convert a trained model into other formats.

    npz          the native format (weights + config + tokenizer copied into one folder)
    int8         per-row symmetric int8 matrices + float32 scales: ~4x smaller; `load_int8()` dequantises for use here
    safetensors  Hugging Face Llama layout (model.layers.N.self_attn.q_proj.weight ... , config.json) written by a tiny pure-Python
                 writer. The architecture is a Llama (RMSNorm, rotate-half RoPE, SwiGLU, tied embeddings) so this is a rename + transpose.

Honest limits: the tensor layout is verified against an independent reference forward pass (tests/test_chat.py) but loading the folder
in `transformers` / llama.cpp has NOT been tried here, and our tokenizer is a custom byte-level BPE, not a Hugging Face `tokenizers`
file - it is exported as tokenizer.json in its own format, so porting a model to another runtime also means porting the tokenizer.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
from typing import Dict, Optional, Tuple

import numpy as np

from .. import fileio
from .model import GPT, GPTConfig
from .paths import Layout

FORMATS = ("npz", "int8", "safetensors")


# ---- int8 ---------------------------------------------------------------------------------------------------------------
def quantize_int8(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in params.items():
        if v.ndim == 2:
            scale = np.maximum(np.abs(v).max(axis=1, keepdims=True), 1e-12) / 127.0
            out[f"q.{k}"] = np.clip(np.rint(v / scale), -127, 127).astype(np.int8)
            out[f"s.{k}"] = scale.astype(np.float32)
        else:
            out[f"f.{k}"] = v.astype(np.float32)
    return out


def dequantize_int8(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    p: Dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        if k.startswith("q."):
            p[k[2:]] = v.astype(np.float32) * arrays["s." + k[2:]]
        elif k.startswith("f."):
            p[k[2:]] = v
    return p


def load_int8(path: str) -> Tuple[GPT, dict]:
    z = fileio.load_npz(path)
    info = json.loads(bytes(z["__meta__"]).decode("utf-8"))
    cfg = GPTConfig.from_dict(info["config"])
    return GPT(cfg, params=dequantize_int8(z)), info.get("meta", {})


# ---- safetensors (format: u64 header length, JSON header, raw little-endian buffers) -------------------------------------
def write_safetensors(path: str, tensors: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None):
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    off = 0
    blobs = []
    dt = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("float64"): "F64", np.dtype("int8"): "I8",
          np.dtype("int64"): "I64", np.dtype("int32"): "I32"}
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        b = a.astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": dt[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * ((8 - len(h) % 8) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    fileio.replace(tmp, path)


def read_safetensors(path: str) -> Dict[str, np.ndarray]:
    rev = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I8": "i1", "I64": "<i8", "I32": "<i4"}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        out = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = np.frombuffer(f.read(b - a), dtype=rev[v["dtype"]]).reshape(v["shape"]).copy()
    return out


def to_llama_tensors(model: GPT) -> Dict[str, np.ndarray]:
    """Our parameters under Hugging Face LlamaForCausalLM names; torch Linear weights are (out, in) so matrices are transposed."""
    p, c = model.p, model.cfg
    if c.n_experts > 1:
        raise ValueError("a mixture-of-experts model has no Llama-format equivalent here (Llama's MLP is dense): export a dense model, or keep the MoE checkpoint")
    t: Dict[str, np.ndarray] = {"model.embed_tokens.weight": p["tok_emb"], "model.norm.weight": p["nf"]}
    for l in range(c.n_layer):
        a, b = f"model.layers.{l}.", f"b{l}."
        t[a + "self_attn.q_proj.weight"] = p[b + "wq"].T
        t[a + "self_attn.k_proj.weight"] = p[b + "wk"].T
        t[a + "self_attn.v_proj.weight"] = p[b + "wv"].T
        t[a + "self_attn.o_proj.weight"] = p[b + "wo"].T
        t[a + "mlp.gate_proj.weight"] = p[b + "wg"].T
        t[a + "mlp.up_proj.weight"] = p[b + "wu"].T
        t[a + "mlp.down_proj.weight"] = p[b + "wd"].T
        t[a + "input_layernorm.weight"] = p[b + "n1"]
        t[a + "post_attention_layernorm.weight"] = p[b + "n2"]
    if not c.tie:
        t["lm_head.weight"] = p["lm_head"]
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in t.items()}


def llama_config(c: GPTConfig, end_id: int, pad_id: int) -> dict:
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": c.d_model, "intermediate_size": c.ffn,
            "num_hidden_layers": c.n_layer, "num_attention_heads": c.n_head, "num_key_value_heads": c.n_head,
            "vocab_size": c.vocab, "max_position_embeddings": c.ctx, "rms_norm_eps": c.eps, "rope_theta": c.rope_theta,
            "tie_word_embeddings": bool(c.tie), "hidden_act": "silu", "attention_bias": False, "mlp_bias": False,
            "torch_dtype": "float32", "eos_token_id": end_id, "pad_token_id": pad_id, "bos_token_id": None}


# ---- driver -------------------------------------------------------------------------------------------------------------------
def export_model(lay: Layout, fmt: str, out_dir: Optional[str] = None, version: Optional[str] = None, cfg=None) -> str:
    """Export the live model (or `version`, a file in models/weights) into a new folder; returns the folder."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format '{fmt}'; choose from {FORMATS}")
    src = os.path.join(lay.weights_dir, os.path.basename(version)) if version else lay.current_weights()
    if not src or not os.path.exists(src):
        raise FileNotFoundError("there is no trained model to export yet (run `zerowhite chat train`)")
    name = os.path.splitext(os.path.basename(src))[0]
    out = out_dir or os.path.join(lay.exports, f"{name}-{fmt}")
    os.makedirs(out, exist_ok=True)
    model, meta = GPT.load(src)
    if os.path.exists(lay.tokenizer_path):
        shutil.copyfile(lay.tokenizer_path, os.path.join(out, "tokenizer.json"))
    if fmt == "npz":
        shutil.copyfile(src, os.path.join(out, "model.npz"))
    elif fmt == "int8":
        arrays = quantize_int8(model.p)
        info = dict(config=model.cfg.to_dict(), meta=dict(meta, quantized="int8-per-row"))
        arrays["__meta__"] = np.frombuffer(json.dumps(info, default=fileio._json_default).encode("utf-8"), np.uint8)
        fileio.save_npz(os.path.join(out, "model.int8.npz"), arrays)
    else:
        from .tokenizer import ChatTemplate
        end_id = pad_id = 0
        if cfg is not None and os.path.exists(lay.tokenizer_path):
            from .tokenizer import Tokenizer
            tpl = ChatTemplate(Tokenizer.load(lay.tokenizer_path), cfg)
            end_id, pad_id = tpl.end_id, tpl.pad_id
        write_safetensors(os.path.join(out, "model.safetensors"), to_llama_tensors(model), {"format": "pt"})
        fileio.write_json(os.path.join(out, "config.json"), llama_config(model.cfg, end_id, pad_id), indent=2)
    fileio.write_json(os.path.join(out, "export.json"), dict(format=fmt, source=os.path.basename(src), meta=meta, params=model.num_params()), indent=2)
    return out
