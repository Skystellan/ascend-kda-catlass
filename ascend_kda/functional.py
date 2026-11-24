# Copyright (c) 2026 Contributors to ascend-kda-catlass.
# Licensed under the MIT License.
"""PyTorch autograd interface for Native-TND AscendC + CATLASS Chunk KDA."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ._jit import load_kda_ops

_META_CACHE: dict[tuple[tuple[int, ...], str], tuple[torch.Tensor, torch.Tensor]] = {}


def build_kda_chunk_meta(
    cu_seqlens: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert cumulative sequence lengths `cu_seqlens` into `(chunk_meta [NC, 4], doc_chunks [N+1])`."""
    if isinstance(cu_seqlens, torch.Tensor):
        cu_host = tuple(int(x) for x in cu_seqlens.detach().cpu().tolist())
    else:
        cu_host = tuple(int(x) for x in cu_seqlens)

    key = (cu_host, str(device))
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached

    meta_rows: list[list[int]] = []
    doc_chunks: list[int] = [0]
    for doc_id in range(len(cu_host) - 1):
        s, e = cu_host[doc_id], cu_host[doc_id + 1]
        local_idx = 0
        pos = s
        while pos < e:
            clen = min(64, e - pos)
            meta_rows.append([doc_id, pos, clen, local_idx])
            pos += clen
            local_idx += 1
        doc_chunks.append(len(meta_rows))

    chunk_meta = torch.tensor(meta_rows, dtype=torch.int32, device=device)
    doc_chunks_t = torch.tensor(doc_chunks, dtype=torch.int32, device=device)
    if len(_META_CACHE) >= 32:
        _META_CACHE.clear()
    _META_CACHE[key] = (chunk_meta, doc_chunks_t)
    return chunk_meta, doc_chunks_t


class _AscendKdaTndFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        cu_host: tuple[int, ...],
        scale: float,
        lower_bound: float,
        eps: float,
        disable_recompute: bool,
    ) -> torch.Tensor:
        ops = load_kda_ops()
        chunk_meta, doc_chunks = build_kda_chunk_meta(cu_host, q.device)
        beta_ht = beta.float().transpose(0, 1).contiguous()
        res = ops.kda_fwd_ascendc(
            q,
            k,
            v,
            g,
            beta_ht,
            a_log,
            dt_bias,
            chunk_meta,
            doc_chunks,
            scale,
            lower_bound,
            eps,
            disable_recompute,
        )
        o_out = res[0]
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.eps = eps
        ctx.disable_recompute = disable_recompute
        if disable_recompute:
            ctx.save_for_backward(
                res[1],
                res[2],
                v,
                g,
                res[3],
                res[4],
                res[5],
                res[6],
                res[7],
                res[8],
                res[9],
                res[10],
                res[11],
                res[12],
                res[13],
                res[14],
                a_log,
                dt_bias,
                chunk_meta,
                doc_chunks,
            )
        else:
            ctx.save_for_backward(q, k, v, g, beta_ht, a_log, dt_bias, chunk_meta, doc_chunks)
        return o_out

    @staticmethod
    def backward(ctx, do_in: torch.Tensor):
        ops = load_kda_ops()
        if ctx.disable_recompute:
            (
                q_hat,
                k_hat,
                v,
                g,
                q_rstd,
                k_rstd,
                beta_eff,
                gk,
                g_last,
                qg,
                kg,
                aqk,
                akk_inv,
                w,
                v_new,
                h_states,
                a_log,
                dt_bias,
                chunk_meta,
                doc_chunks,
            ) = ctx.saved_tensors
        else:
            q, k, v, g, beta_ht, a_log, dt_bias, chunk_meta, doc_chunks = ctx.saved_tensors
            res = ops.kda_fwd_ascendc(
                q,
                k,
                v,
                g,
                beta_ht,
                a_log,
                dt_bias,
                chunk_meta,
                doc_chunks,
                ctx.scale,
                ctx.lower_bound,
                ctx.eps,
                True,
            )
            (
                _,
                q_hat,
                k_hat,
                q_rstd,
                k_rstd,
                beta_eff,
                gk,
                g_last,
                qg,
                kg,
                aqk,
                akk_inv,
                w,
                v_new,
                h_states,
            ) = res

        dq, dk, dv, dg, dbeta_ht, da_log, ddt_bias = ops.kda_bwd_ascendc(
            do_in,
            q_hat,
            k_hat,
            v,
            g,
            q_rstd,
            k_rstd,
            beta_eff,
            gk,
            g_last,
            qg,
            kg,
            aqk,
            akk_inv,
            w,
            v_new,
            h_states,
            a_log,
            dt_bias,
            chunk_meta,
            doc_chunks,
            ctx.scale,
            ctx.lower_bound,
        )
        dbeta = dbeta_ht.transpose(0, 1).contiguous()
        return dq, dk, dv, dg, dbeta, da_log, ddt_bias, None, None, None, None, None


def chunk_kda_tnd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: Sequence[int] | torch.Tensor,
    *,
    scale: float | None = None,
    lower_bound: float = -5.0,
    eps: float = 1e-6,
    disable_recompute: bool = True,
) -> torch.Tensor:
    """Native-TND packed variable-length KDA forward + backward on Ascend 910C (Atlas A2).

    Fuses exact FP32 Q/K L2Norm, `safe_gate` (`lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`),
    and `beta_eff = sigmoid(beta)` directly inside the AscendC + CATLASS kernels with zero
    `TND <-> NTD` transposes.

    Args:
        q, k, v, g: BF16 tensors of shape `[T, H, 128]`.
        beta: FP32 or BF16 raw beta logits of shape `[T, H]`.
        a_log: FP32 tensor of shape `[H]`.
        dt_bias: FP32 tensor of shape `[H, 128]` or `[H * 128]`.
        cu_seqlens: Cumulative sequence offsets `[0, s_1, ..., T]`.
        scale: Query scale factor (defaults to `128 ** -0.5`).
        lower_bound: Safe-gate lower bound (default `-5.0`).
        eps: L2Norm epsilon (default `1e-6`).
        disable_recompute: If `True`, saves forward chunk states for backward; if `False`, recomputes in backward.

    Returns:
        Output BF16 tensor of shape `[T, H, 128]`.
    """
    if isinstance(cu_seqlens, torch.Tensor):
        cu_host = tuple(int(x) for x in cu_seqlens.detach().cpu().tolist())
    else:
        cu_host = tuple(int(x) for x in cu_seqlens)
    eff_scale = float(q.shape[-1] ** -0.5) if scale is None else float(scale)
    return _AscendKdaTndFn.apply(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        cu_host,
        eff_scale,
        float(lower_bound),
        float(eps),
        bool(disable_recompute),
    )
