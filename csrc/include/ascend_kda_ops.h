// Copyright (c) 2026 Contributors to ascend-kda-catlass.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ascend_kda {

struct KdaFwdPrepHostParams {
    uint32_t total_tokens;
    uint32_t heads;
    uint32_t num_chunks;
    float scale;
    float lower_bound;
    float eps;
    uint64_t q_raw;
    uint64_t k_raw;
    uint64_t v_raw;
    uint64_t a_raw;
    uint64_t beta_raw;
    uint64_t a_log;
    uint64_t dt_bias;
    uint64_t chunk_meta;
    uint64_t q_hat;
    uint64_t k_hat;
    uint64_t q_rstd;
    uint64_t k_rstd;
    uint64_t beta_eff;
    uint64_t gk;
    uint64_t g_last;
    uint64_t qg;
    uint64_t kg;
    uint64_t k_beta_g;
    uint64_t v_beta;
    uint64_t aqk_diag;
    uint64_t akk_diag;
};

struct KdaFwdPrepIntraWyHostParams {
    uint32_t total_tokens;
    uint32_t heads;
    uint32_t num_chunks;
    float scale;
    float lower_bound;
    float eps;
    uint64_t q_raw;
    uint64_t k_raw;
    uint64_t v_raw;
    uint64_t a_raw;
    uint64_t beta_raw;
    uint64_t a_log;
    uint64_t dt_bias;
    uint64_t chunk_meta;
    uint64_t q_hat;
    uint64_t k_hat;
    uint64_t q_rstd;
    uint64_t k_rstd;
    uint64_t beta_eff;
    uint64_t gk;
    uint64_t g_last;
    uint64_t qg;
    uint64_t kg;
    uint64_t aqk_out;
    uint64_t akk_inv_out;
    uint64_t w_out;
    uint64_t u_out;
    uint64_t workspace;
};

struct KdaBwdFinHostParams {
    uint32_t total_tokens;
    uint32_t heads;
    uint32_t num_chunks;
    float scale;
    float lower_bound;
    uint64_t chunk_meta;
    uint64_t q_hat;
    uint64_t k_hat;
    uint64_t v_raw;
    uint64_t v_new;
    uint64_t a_raw;
    uint64_t gk;
    uint64_t g_last;
    uint64_t beta_eff;
    uint64_t q_rstd;
    uint64_t k_rstd;
    uint64_t a_log;
    uint64_t dt_bias;
    uint64_t akk_inv;
    uint64_t h;
    uint64_t dh;
    uint64_t do_in;
    uint64_t dv_in;
    uint64_t daqk_in;
    uint64_t dgk_last_in;
    uint64_t dq_out;
    uint64_t dk_out;
    uint64_t dv_out;
    uint64_t da_out;
    uint64_t dbeta_out;
    uint64_t da_log_part;
    uint64_t ddt_bias_part;
    uint64_t workspace;
};

size_t kda_fwd_intra_wy_workspace_bytes();
size_t kda_fwd_ho_workspace_bytes();
size_t kda_bwd_dhu_workspace_bytes();
size_t kda_bwd_finalize_workspace_bytes();

void launch_kda_fwd_prep(const KdaFwdPrepHostParams& hp, void* stream);

void launch_kda_fwd_prep_intra_wy(
    const KdaFwdPrepIntraWyHostParams& hp, uint64_t sync_address, void* stream);

void launch_kda_fwd_ho(
    uint32_t total_tokens, uint32_t heads, uint32_t num_docs, uint32_t save_states, float scale,
    const void* doc_chunks, const void* chunk_meta,
    const void* w, const void* u, const void* qg, const void* kg, const void* aqk, const void* g_last,
    void* o_out, void* v_new_out, void* h_out,
    void* workspace, uint64_t sync_address, void* stream);

void launch_kda_bwd_dhu(
    uint32_t total_tokens, uint32_t heads, uint32_t num_docs, float scale,
    const void* doc_chunks, const void* chunk_meta,
    const void* qg, const void* kg, const void* w, const void* v_new, const void* aqk,
    const void* g_last, const void* do_in, const void* h_in,
    void* dh_out, void* dv_out, void* daqk_out, void* dgk_last_out,
    void* workspace, uint64_t sync_address, void* stream);

void launch_kda_bwd_finalize(const KdaBwdFinHostParams& hp, uint64_t sync_address, void* stream);

}  // namespace ascend_kda
