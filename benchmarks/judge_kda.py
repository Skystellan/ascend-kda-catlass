# Copyright (c) 2026 Contributors to ascend-kda-catlass.
# Licensed under the MIT License.
"""Interleaved benchmark & Level-1 PipeUtilization profiler for ascend-kda-catlass."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from ascend_kda import build_kda_chunk_meta, load_kda_ops


def bench_ms(fn, warmup: int = 4, iters: int = 15) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(1000.0 * (time.perf_counter() - t0))
    return {
        "median_ms": float(statistics.median(ts)),
        "mean_ms": float(statistics.mean(ts)),
        "min_ms": float(min(ts)),
    }


def parse_level1_profile(prof_dir: Path) -> dict:
    csvs = sorted(prof_dir.glob("**/kernel_details.csv"))
    if not csvs:
        return {}
    by_k: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])
    total = 0.0

    def fv(x) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    with csvs[-1].open("r", newline="") as f:
        for row in csv.DictReader(f):
            dur = fv(row.get("Duration(us)")) / 1000.0
            if dur <= 0.0:
                continue
            k = row.get("Type") or row.get("Name") or "unknown"
            for short in ("KdaFwdPrepIntraWy", "KdaFwdHO", "KdaBwdDhu", "KdaBwdFinalize"):
                if short in k or short in (row.get("Name") or ""):
                    k = short
                    break
            total += dur
            by_k[k][0] += 1
            by_k[k][1] += dur
            by_k[k][2] += dur * fv(row.get("aic_mac_ratio"))
            by_k[k][3] += dur * fv(row.get("aic_mte2_ratio"))
            by_k[k][4] += dur * fv(row.get("aiv_vec_ratio"))
            by_k[k][5] += dur * fv(row.get("aiv_mte2_ratio"))

    summary = {"total_kernel_ms": round(total, 4), "kernels": {}}
    print(f"\n=== Level-1 PipeUtilization Kernel Profile (total={total:.3f} ms) ===")
    for k, (cnt, dur, mac, am2, vec, vm2) in sorted(by_k.items(), key=lambda x: -x[1][1])[:12]:
        rec = {
            "cnt": cnt,
            "ms": round(dur, 4),
            "mac": round(mac / dur, 3),
            "amte2": round(am2 / dur, 3),
            "vec": round(vec / dur, 3),
            "vmte2": round(vm2 / dur, 3),
        }
        summary["kernels"][k] = rec
        print(
            f"  {k:24s}: cnt={cnt:2d} ms={dur:6.3f} "
            f"mac={rec['mac']:.3f} aMTE2={rec['amte2']:.3f} vec={rec['vec']:.3f} vMTE2={rec['vmte2']:.3f}"
        )
    return summary


def main() -> None:
    import torch_npu

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    ops = load_kda_ops()

    T, H, D = args.tokens, args.heads, 128
    cu = tuple(range(0, T, args.seq_len)) + (T,)
    torch.manual_seed(100)
    q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16) * 0.5
    k = torch.randn(T, H, D, device=device, dtype=torch.bfloat16) * 0.5
    v = torch.randn(T, H, D, device=device, dtype=torch.bfloat16) * 0.5
    g = torch.randn(T, H, D, device=device, dtype=torch.bfloat16) * 0.5
    beta = torch.randn(T, H, device=device, dtype=torch.float32) * 0.5
    a_log = torch.randn(H, device=device, dtype=torch.float32) * 0.2
    dt_bias = torch.randn(H * D, device=device, dtype=torch.float32) * 0.3 - 1.5
    do = torch.randn(T, H, D, device=device, dtype=torch.bfloat16) * 0.5
    cm, dc = build_kda_chunk_meta(cu, device)
    bht = beta.t().contiguous()
    scale = D ** -0.5

    def run_fwd():
        return ops.kda_fwd_ascendc(q, k, v, g, bht, a_log, dt_bias, cm, dc, scale, -5.0, 1e-6, True)

    cached = run_fwd()
    torch.npu.synchronize()

    def run_bwd():
        return ops.kda_bwd_ascendc(
            do,
            cached[1],
            cached[2],
            v,
            g,
            cached[3],
            cached[4],
            cached[5],
            cached[6],
            cached[7],
            cached[8],
            cached[9],
            cached[10],
            cached[11],
            cached[12],
            cached[13],
            cached[14],
            a_log,
            dt_bias,
            cm,
            dc,
            scale,
            -5.0,
        )

    def run_all():
        f = run_fwd()
        ops.kda_bwd_ascendc(
            do,
            f[1],
            f[2],
            v,
            g,
            f[3],
            f[4],
            f[5],
            f[6],
            f[7],
            f[8],
            f[9],
            f[10],
            f[11],
            f[12],
            f[13],
            f[14],
            a_log,
            dt_bias,
            cm,
            dc,
            scale,
            -5.0,
        )

    t_fwd = bench_ms(run_fwd, warmup=args.warmup, iters=args.iters)
    t_bwd = bench_ms(run_bwd, warmup=args.warmup, iters=args.iters)
    t_tot = bench_ms(run_all, warmup=args.warmup, iters=args.iters)

    res = {
        "shape": {"T": T, "H": H, "D": D, "cu_seqlens": list(cu)},
        "fwd_ms": round(t_fwd["median_ms"], 4),
        "bwd_ms": round(t_bwd["median_ms"], 4),
        "total_ms": round(t_tot["median_ms"], 4),
    }
    print(f"T={T}, H={H}, D={D}: fwd={res['fwd_ms']:.3f} ms | bwd={res['bwd_ms']:.3f} ms | total={res['total_ms']:.3f} ms")

    if args.profile:
        prof_dir = Path("prof_ascend_kda")
        prof_dir.mkdir(parents=True, exist_ok=True)
        ec = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=ec,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(prof_dir)),
        ):
            run_all()
            torch.npu.synchronize()
        res["profile"] = parse_level1_profile(prof_dir)

    if args.output is not None:
        args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
