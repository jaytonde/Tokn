
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class InferenceContext:
    is_prefill: bool
    cu_seqlens_q: Optional[Any] = None
    cu_seqlens_k: Optional[Any] = None
    max_seqlen_q: Optional[int] = None
    max_seqlen_k: Optional[int] = None
    slot_mapping: Optional[Any] = None
    context_lens: Optional[Any] = None
    block_tables: Optional[Any] = None
    # Decode only: True -> flash attention uses num_splits=1 (single-pass
    # kernel, bit-identical for seqlen <= 256, ~7us/call faster); False ->
    # splitkv path (reference behavior for seqlen > 256).
    use_fast_flash: bool = False


_CONTEXT: Optional[InferenceContext] = None


def set_context(
    *,
    is_prefill: bool,
    cu_seqlens_q: Optional[Any] = None,
    cu_seqlens_k: Optional[Any] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    slot_mapping: Optional[Any] = None,
    context_lens: Optional[Any] = None,
    block_tables: Optional[Any] = None,
    use_fast_flash: bool = False,
) -> None:
    global _CONTEXT
    _CONTEXT = InferenceContext(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        use_fast_flash=use_fast_flash,
    )


def get_context() -> InferenceContext:
    if _CONTEXT is None:
        raise RuntimeError("Inference context is not set. Call set_context() before model forward.")
    return _CONTEXT


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = None
    