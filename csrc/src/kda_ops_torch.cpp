// Copyright (c) 2026 Contributors to ascend-kda-catlass.
// Licensed under the MIT License.

#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <algorithm>
#include <vector>

#include "ascend_kda_ops.h"

namespace {

at::Tensor get_kda_workspace(const at::Tensor& ref) {
    const size_t max_ws = std::max(
        std::max(ascend_kda::kda_fwd_intra_wy_workspace_bytes(), ascend_kda::kda_fwd_ho_workspace_bytes()),
        std::max(ascend_kda::kda_bwd_dhu_workspace_bytes(), ascend_kda::kda_bwd_finalize_workspace_bytes()));
    const int64_t ws_bytes = static_cast<int64_t>(max_ws);
    static at::Tensor kda_ws_cache[16];
    int dev_idx = ref.device().index();
    if (dev_idx < 0 || dev_idx >= 16) {
        dev_idx = 0;
    }
    if (!kda_ws_cache[dev_idx].defined() || kda_ws_cache[dev_idx].numel() < ws_bytes ||
        kda_ws_cache[dev_idx].device() != ref.device()) {
        kda_ws_cache[dev_idx] = at::empty({ws_bytes}, ref.options().dtype(at::kByte));
        kda_ws_cache[dev_idx].narrow(0, 0, 65536).zero_();
    }
    return kda_ws_cache[dev_idx];
}

std::vector<at::Tensor> kda_fwd_ascendc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& beta_ht,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& chunk_meta,
    const at::Tensor& doc_chunks,
    double scale,
    double lower_bound,
    double eps,
    bool save_states) {
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && a.dim() == 3,
                "kda_fwd_ascendc expects 3D TND [T, H, 128] q, k, v, a");
    TORCH_CHECK(q.size(2) == 128 && k.sizes() == q.sizes() && v.sizes() == q.sizes() && a.sizes() == q.sizes(),
                "kda_fwd_ascendc requires matching [T, H, 128] shapes");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kBFloat16,
                "kda_fwd_ascendc expects BF16 q, k, v, a");
    const int64_t T = q.size(0);
    const int64_t H = q.size(1);
    const int64_t NC = chunk_meta.size(0);
    const int64_t num_docs = doc_chunks.numel() - 1;
    TORCH_CHECK(chunk_meta.dim() == 2 && chunk_meta.size(1) == 4 && chunk_meta.scalar_type() == at::kInt,
                "kda_fwd_ascendc expects int32 chunk_meta [NC, 4]");
    TORCH_CHECK(doc_chunks.dim() == 1 && num_docs >= 1 && doc_chunks.scalar_type() == at::kInt,
                "kda_fwd_ascendc expects int32 doc_chunks [num_docs + 1]");
    TORCH_CHECK(beta_ht.dim() == 2 && beta_ht.size(0) == H && beta_ht.size(1) == T,
                "kda_fwd_ascendc expects beta_ht [H, T]");

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto a_c = a.contiguous();
    auto beta_c = (beta_ht.scalar_type() == at::kFloat) ? beta_ht.contiguous() : beta_ht.to(at::kFloat).contiguous();
    auto alog_c = (a_log.scalar_type() == at::kFloat) ? a_log.reshape({H}).contiguous() : a_log.to(at::kFloat).reshape({H}).contiguous();
    auto dtb_c = (dt_bias.scalar_type() == at::kFloat) ? dt_bias.reshape({H, 128}).contiguous() : dt_bias.to(at::kFloat).reshape({H, 128}).contiguous();
    auto meta_c = chunk_meta.contiguous();
    auto doc_c = doc_chunks.contiguous();

    auto bf_opts = q_c.options().dtype(at::kBFloat16);
    auto f32_opts = q_c.options().dtype(at::kFloat);

    auto o_out = at::empty({T, H, 128}, bf_opts);
    auto q_hat = at::empty({T, H, 128}, bf_opts);
    auto k_hat = at::empty({T, H, 128}, bf_opts);
    auto q_rstd = at::empty({H, T}, f32_opts);
    auto k_rstd = at::empty({H, T}, f32_opts);
    auto beta_eff = at::empty({H, T}, f32_opts);
    auto gk = at::empty({T, H, 128}, f32_opts);
    auto g_last = at::empty({NC, H, 128}, f32_opts);
    auto qg = at::empty({T, H, 128}, bf_opts);
    auto kg = at::empty({T, H, 128}, bf_opts);
    auto aqk = at::empty({NC, H, 64, 64}, bf_opts);
    auto akk_inv = at::empty({NC, H, 64, 64}, bf_opts);
    auto w = at::empty({T, H, 128}, bf_opts);
    auto u = at::empty({T, H, 128}, bf_opts);
    at::Tensor v_new = save_states ? at::empty({T, H, 128}, bf_opts) : at::empty({0}, bf_opts);
    at::Tensor h_states = save_states ? at::empty({NC, H, 128, 128}, bf_opts) : at::empty({0}, bf_opts);

    if (T == 0 || NC == 0) {
        return {o_out, q_hat, k_hat, q_rstd, k_rstd, beta_eff, gk, g_last, qg, kg, aqk, akk_inv, w, v_new, h_states};
    }

    auto workspace = get_kda_workspace(q_c);
    const float scale_f = static_cast<float>(scale);
    const float lb_f = static_cast<float>(lower_bound);
    const float eps_f = static_cast<float>(eps);
    const uint32_t save_u = save_states ? 1u : 0u;

    ascend_kda::KdaFwdPrepIntraWyHostParams hp{
        static_cast<uint32_t>(T), static_cast<uint32_t>(H), static_cast<uint32_t>(NC),
        scale_f, lb_f, eps_f,
        reinterpret_cast<uint64_t>(q_c.data_ptr()),
        reinterpret_cast<uint64_t>(k_c.data_ptr()),
        reinterpret_cast<uint64_t>(v_c.data_ptr()),
        reinterpret_cast<uint64_t>(a_c.data_ptr()),
        reinterpret_cast<uint64_t>(beta_c.data_ptr()),
        reinterpret_cast<uint64_t>(alog_c.data_ptr()),
        reinterpret_cast<uint64_t>(dtb_c.data_ptr()),
        reinterpret_cast<uint64_t>(meta_c.data_ptr()),
        reinterpret_cast<uint64_t>(q_hat.data_ptr()),
        reinterpret_cast<uint64_t>(k_hat.data_ptr()),
        reinterpret_cast<uint64_t>(q_rstd.data_ptr()),
        reinterpret_cast<uint64_t>(k_rstd.data_ptr()),
        reinterpret_cast<uint64_t>(beta_eff.data_ptr()),
        reinterpret_cast<uint64_t>(gk.data_ptr()),
        reinterpret_cast<uint64_t>(g_last.data_ptr()),
        reinterpret_cast<uint64_t>(qg.data_ptr()),
        reinterpret_cast<uint64_t>(kg.data_ptr()),
        reinterpret_cast<uint64_t>(aqk.data_ptr()),
        reinterpret_cast<uint64_t>(akk_inv.data_ptr()),
        reinterpret_cast<uint64_t>(w.data_ptr()),
        reinterpret_cast<uint64_t>(u.data_ptr()),
        reinterpret_cast<uint64_t>(workspace.data_ptr())
    };

    std::vector<at::Tensor> held_fwd = {
        q_c, k_c, v_c, a_c, beta_c, alog_c, dtb_c, meta_c, doc_c,
        o_out, q_hat, k_hat, q_rstd, k_rstd, beta_eff, gk, g_last,
        qg, kg, aqk, akk_inv, w, u, v_new, h_states, workspace
    };

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand::RunOpApiV2(
        "ascend_kda_fwd", [=]() -> int {
            (void)held_fwd;
            uint64_t sync_address = 0;
            auto code = aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&sync_address));
            TORCH_CHECK(code == ACL_SUCCESS, "aclrtGetHardwareSyncAddr failed: ", code);
            ascend_kda::launch_kda_fwd_prep_intra_wy(hp, sync_address, stream);
            ascend_kda::launch_kda_fwd_ho(
                static_cast<uint32_t>(T), static_cast<uint32_t>(H), static_cast<uint32_t>(num_docs), save_u, scale_f,
                doc_c.data_ptr(), meta_c.data_ptr(),
                w.data_ptr(), u.data_ptr(), qg.data_ptr(), kg.data_ptr(), aqk.data_ptr(), g_last.data_ptr(),
                o_out.data_ptr(),
                save_states ? v_new.data_ptr() : nullptr,
                save_states ? h_states.data_ptr() : nullptr,
                workspace.data_ptr(), sync_address, stream);
            return 0;
        });

    return {o_out, q_hat, k_hat, q_rstd, k_rstd, beta_eff, gk, g_last, qg, kg, aqk, akk_inv, w, v_new, h_states};
}

std::vector<at::Tensor> kda_bwd_ascendc(
    const at::Tensor& do_in,
    const at::Tensor& q_hat,
    const at::Tensor& k_hat,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& q_rstd,
    const at::Tensor& k_rstd,
    const at::Tensor& beta_eff,
    const at::Tensor& gk,
    const at::Tensor& g_last,
    const at::Tensor& qg,
    const at::Tensor& kg,
    const at::Tensor& aqk,
    const at::Tensor& akk_inv,
    const at::Tensor& w,
    const at::Tensor& v_new,
    const at::Tensor& h_states,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& chunk_meta,
    const at::Tensor& doc_chunks,
    double scale,
    double lower_bound) {
    auto do_c = do_in.contiguous();
    auto qh_c = q_hat.contiguous();
    auto kh_c = k_hat.contiguous();
    auto v_c = v.contiguous();
    auto a_c = a.contiguous();
    auto qr_c = q_rstd.contiguous();
    auto kr_c = k_rstd.contiguous();
    auto be_c = beta_eff.contiguous();
    auto gk_c = gk.contiguous();
    auto gl_c = g_last.contiguous();
    auto qg_c = qg.contiguous();
    auto kg_c = kg.contiguous();
    auto aqk_c = aqk.contiguous();
    auto akki_c = akk_inv.contiguous();
    auto w_c = w.contiguous();
    auto vn_c = v_new.contiguous();
    auto h_c = h_states.contiguous();
    auto meta_c = chunk_meta.contiguous();
    auto doc_c = doc_chunks.contiguous();

    const int64_t T = qh_c.size(0);
    const int64_t H = qh_c.size(1);
    const int64_t NC = meta_c.size(0);
    const int64_t num_docs = doc_c.numel() - 1;

    auto alog_c = (a_log.scalar_type() == at::kFloat) ? a_log.reshape({H}).contiguous() : a_log.to(at::kFloat).reshape({H}).contiguous();
    auto dtb_c = (dt_bias.scalar_type() == at::kFloat) ? dt_bias.reshape({H, 128}).contiguous() : dt_bias.to(at::kFloat).reshape({H, 128}).contiguous();

    auto bf_opts = qh_c.options().dtype(at::kBFloat16);
    auto f32_opts = qh_c.options().dtype(at::kFloat);

    auto dq = at::empty({T, H, 128}, bf_opts);
    auto dk = at::empty({T, H, 128}, bf_opts);
    auto dv = at::empty({T, H, 128}, bf_opts);
    auto da = at::empty({T, H, 128}, bf_opts);
    auto dbeta_ht = at::empty({H, T}, f32_opts);
    auto da_log_part = at::zeros({48, H}, f32_opts);
    auto ddt_bias_part = at::zeros({48, H, 128}, f32_opts);

    if (T == 0 || NC == 0) {
        auto da_log = at::zeros_like(a_log);
        auto ddt_bias = at::zeros_like(dt_bias);
        return {dq, dk, dv, da, dbeta_ht, da_log, ddt_bias};
    }

    auto dh_states = at::empty({NC, H, 128, 128}, bf_opts);
    auto dv_new = at::empty({T, H, 128}, bf_opts);
    auto daqk = at::empty({NC, H, 64, 64}, f32_opts);
    auto dgk_last_h = at::empty({NC, H, 128}, f32_opts);

    auto workspace = get_kda_workspace(qh_c);
    const float scale_f = static_cast<float>(scale);
    const float lb_f = static_cast<float>(lower_bound);

    ascend_kda::KdaBwdFinHostParams hp{
        static_cast<uint32_t>(T), static_cast<uint32_t>(H), static_cast<uint32_t>(NC),
        scale_f, lb_f,
        reinterpret_cast<uint64_t>(meta_c.data_ptr()),
        reinterpret_cast<uint64_t>(qh_c.data_ptr()),
        reinterpret_cast<uint64_t>(kh_c.data_ptr()),
        reinterpret_cast<uint64_t>(v_c.data_ptr()),
        reinterpret_cast<uint64_t>(vn_c.data_ptr()),
        reinterpret_cast<uint64_t>(a_c.data_ptr()),
        reinterpret_cast<uint64_t>(gk_c.data_ptr()),
        reinterpret_cast<uint64_t>(gl_c.data_ptr()),
        reinterpret_cast<uint64_t>(be_c.data_ptr()),
        reinterpret_cast<uint64_t>(qr_c.data_ptr()),
        reinterpret_cast<uint64_t>(kr_c.data_ptr()),
        reinterpret_cast<uint64_t>(alog_c.data_ptr()),
        reinterpret_cast<uint64_t>(dtb_c.data_ptr()),
        reinterpret_cast<uint64_t>(akki_c.data_ptr()),
        reinterpret_cast<uint64_t>(h_c.data_ptr()),
        reinterpret_cast<uint64_t>(dh_states.data_ptr()),
        reinterpret_cast<uint64_t>(do_c.data_ptr()),
        reinterpret_cast<uint64_t>(dv_new.data_ptr()),
        reinterpret_cast<uint64_t>(daqk.data_ptr()),
        reinterpret_cast<uint64_t>(dgk_last_h.data_ptr()),
        reinterpret_cast<uint64_t>(dq.data_ptr()),
        reinterpret_cast<uint64_t>(dk.data_ptr()),
        reinterpret_cast<uint64_t>(dv.data_ptr()),
        reinterpret_cast<uint64_t>(da.data_ptr()),
        reinterpret_cast<uint64_t>(dbeta_ht.data_ptr()),
        reinterpret_cast<uint64_t>(da_log_part.data_ptr()),
        reinterpret_cast<uint64_t>(ddt_bias_part.data_ptr()),
        reinterpret_cast<uint64_t>(workspace.data_ptr())
    };

    std::vector<at::Tensor> held_bwd = {
        do_c, qh_c, kh_c, v_c, a_c, qr_c, kr_c, be_c, gk_c, gl_c, qg_c, kg_c,
        aqk_c, akki_c, w_c, vn_c, h_c, alog_c, dtb_c, meta_c, doc_c,
        dh_states, dv_new, daqk, dgk_last_h, dq, dk, dv, da, dbeta_ht, da_log_part, ddt_bias_part, workspace
    };

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand::RunOpApiV2(
        "ascend_kda_bwd", [=]() -> int {
            (void)held_bwd;
            uint64_t sync_address = 0;
            auto code = aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&sync_address));
            TORCH_CHECK(code == ACL_SUCCESS, "aclrtGetHardwareSyncAddr failed: ", code);
            ascend_kda::launch_kda_bwd_dhu(
                static_cast<uint32_t>(T), static_cast<uint32_t>(H), static_cast<uint32_t>(num_docs), scale_f,
                doc_c.data_ptr(), meta_c.data_ptr(),
                qg_c.data_ptr(), kg_c.data_ptr(), w_c.data_ptr(), vn_c.data_ptr(), aqk_c.data_ptr(),
                gl_c.data_ptr(), do_c.data_ptr(), h_c.data_ptr(),
                dh_states.data_ptr(), dv_new.data_ptr(), daqk.data_ptr(), dgk_last_h.data_ptr(),
                workspace.data_ptr(), sync_address, stream);
            ascend_kda::launch_kda_bwd_finalize(hp, sync_address, stream);
            return 0;
        });

    auto da_log = da_log_part.sum(0).reshape(a_log.sizes()).to(a_log.scalar_type());
    auto ddt_bias = ddt_bias_part.sum(0).reshape(dt_bias.sizes()).to(dt_bias.scalar_type());
    return {dq, dk, dv, da, dbeta_ht, da_log, ddt_bias};
}

}  // namespace

TORCH_LIBRARY(ascend_kda_ops, m) {
    m.def("kda_fwd_ascendc(Tensor q, Tensor k, Tensor v, Tensor a, Tensor beta_ht, Tensor a_log, Tensor dt_bias, Tensor chunk_meta, Tensor doc_chunks, float scale, float lower_bound, float eps, bool save_states) -> Tensor[]");
    m.def("kda_bwd_ascendc(Tensor do_in, Tensor q_hat, Tensor k_hat, Tensor v, Tensor a, Tensor q_rstd, Tensor k_rstd, Tensor beta_eff, Tensor gk, Tensor g_last, Tensor qg, Tensor kg, Tensor aqk, Tensor akk_inv, Tensor w, Tensor v_new, Tensor h_states, Tensor a_log, Tensor dt_bias, Tensor chunk_meta, Tensor doc_chunks, float scale, float lower_bound) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(ascend_kda_ops, PrivateUse1, m) {
    m.impl("kda_fwd_ascendc", kda_fwd_ascendc);
    m.impl("kda_bwd_ascendc", kda_bwd_ascendc);
}
