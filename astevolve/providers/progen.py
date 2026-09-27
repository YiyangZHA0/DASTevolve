

from __future__ import annotations
import gc
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer

from astevolve.runtime.paths import model_path


_DEFAULT_MODEL_DIR = Path(
    os.environ.get("ASTEVOLVE_PROGEN_MODEL_DIR", str(model_path("progen2-small")))
)
_PROGEN_DEVICE_ENV = "ASTEVOLVE_PROGEN_DEVICE"
_MODEL = None
_TOKENIZER = None
_CACHE: Dict[Tuple[str, str], Dict[str, float]] = {}


def clear_progen_model_cache(*, clear_score_cache: bool = False) -> None:

    global _MODEL, _TOKENIZER
    try:
        if _MODEL is not None and hasattr(_MODEL, "cpu"):
            _MODEL.cpu()
    except Exception:
        pass
    _MODEL = None
    _TOKENIZER = None
    if clear_score_cache:
        _CACHE.clear()
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _patch_progen_transformer_compat(model) -> None:
    transformer = getattr(model, "transformer", None)
    if transformer is None or hasattr(transformer, "get_head_mask"):
        return

    def _get_head_mask(head_mask, num_hidden_layers, is_attention_chunked=False):
        del is_attention_chunked
        if head_mask is None:
            return [None] * int(num_hidden_layers)
        return head_mask

    transformer.get_head_mask = _get_head_mask


def _patch_progen_runtime_tensors(model, device: torch.device) -> None:
    for module in model.modules():
        if not hasattr(module, "scale_attn") or not hasattr(module, "head_dim"):
            continue
        module.scale_attn = torch.sqrt(
            torch.tensor(module.head_dim, dtype=torch.float32, device=device)
        )


def _get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    configured = str(os.environ.get(_PROGEN_DEVICE_ENV, "") or "").strip()
    if configured:
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer(device: Optional[str] = None):


    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            str(_DEFAULT_MODEL_DIR),
            trust_remote_code=True,
            local_files_only=True,
        )
        _patch_progen_transformer_compat(_MODEL)
        _TOKENIZER = AutoTokenizer.from_pretrained(
            str(_DEFAULT_MODEL_DIR),
            trust_remote_code=True,
            local_files_only=True,

        )

    dev = _get_device(device)
    _MODEL = _MODEL.to(dev)
    _patch_progen_runtime_tensors(_MODEL, dev)
    _MODEL.eval()
    return _MODEL, _TOKENIZER


@torch.no_grad()
def sequence_loglikelihood(
    seq: str,
    device: Optional[str] = None,
    prepend_token: Optional[str] = "1"
) -> Dict[str, float]:


    if not seq:
        return {"loglik_sum": 0.0, "loglik_avg": 0.0}

    if prepend_token:
        if not seq.startswith(prepend_token):
            seq = prepend_token + seq + "2"

    dev = _get_device(device)
    cache_key = (seq, str(dev))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    model, tokenizer = get_model_and_tokenizer(device=str(dev))


    ids = tokenizer.encode(seq)
    if len(ids) < 2:
        out = {"loglik_sum": 0.0, "loglik_avg": 0.0}
        _CACHE[cache_key] = out
        return out

    input_ids = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
    logits = model(input_ids).logits

    pred_logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_loglik = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    loglik_sum = float(token_loglik.sum().item())
    loglik_avg = float(token_loglik.mean().item())

    out = {"loglik_sum": loglik_sum, "loglik_avg": loglik_avg}
    _CACHE[cache_key] = out
    return out
