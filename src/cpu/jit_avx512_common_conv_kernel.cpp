/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx512_common_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)
#define KMASK_FLCONV   k6 /* Note: Inside kernel, Opmask k6 is
                             reserved for ver_fma version of:
                             - FWD:   L(ast conv)
                             - BWD-D: F(irst conv)
                             - BWD-F: L(ast conv)
                           */
#define SET_FLCONV_KMASK(m)                                                    \
do {                                                                           \
    Opmask _mask = KMASK_FLCONV;                                               \
    push(Xbyak::Reg64(reg_tmp));                                               \
    Reg32 reg_tmp_w = reg_tmp.cvt32();                                         \
    mov(reg_tmp_w, (1 << m) - 1);                                              \
    kmovw(_mask, reg_tmp_w);                                                   \
    pop(Xbyak::Reg64(reg_tmp));                                                \
} while(0)

#define CACHE_LINE_SHIFT  6 /* cache line size: 64 */
#define IF_NEW_CACHE_LINE(old_cache_line, off, prf_stmt)                       \
do {                                                                           \
    int _new_cache_line = (off) >> CACHE_LINE_SHIFT;                           \
    if (old_cache_line != _new_cache_line) {                                   \
        old_cache_line = _new_cache_line;                                      \
        prf_stmt;                                                              \
    }                                                                          \
} while(0)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {

constexpr auto small_spatial = 14;

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(jcp.prop_kind,
                forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;
    switch (jcp.ver) {
    case ver_fma:
        jcp.loop_order = loop_cgn;
    case ver_4vnni:
    case ver_4fma:
        jcp.loop_order
            = (w <= small_spatial && h <= small_spatial) ? loop_cgn : loop_gnc;
        break;
    default:
        assert(!"unsupported convolution version");
    }
}

}

void jit_avx512_common_conv_fwd_kernel::prepare_output(int ur_w)
{
    int saved_cache_line = -1;
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
            int aux_output_offset = get_output_offset(j, k);
            IF_NEW_CACHE_LINE(saved_cache_line, aux_output_offset,
                              mic_prefetcht1(EVEX_compress_addr(
                                      reg_out_prf, aux_output_offset)));
        }
}

void jit_avx512_common_conv_fwd_kernel::store_output(int ur_w)
{
    Label no_update_label, store_label, relu_label;

    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    }
    cmp(reg_channel, 0);
    je(no_update_label, T_NEAR);

    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = jcp.is_lastconv
                ? zmm_out(j, k) | KMASK_FLCONV : zmm_out(j, k);
            int aux_output_offset = get_output_offset(j, k);
            vadd(zmm, reg_out, aux_output_offset);
        }
    jmp(relu_label, T_NEAR);

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_out * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = jcp.is_lastconv
                    ? zmm_out(j, k) | KMASK_FLCONV : zmm_out(j, k);
                vadd(zmm, reg_bias, bias_offset);
            }
            mic_prefetcht1(EVEX_compress_addr(reg_bias, bias_offset + 64));
        }
    }

    L(relu_label);
    if (jcp.with_relu) {
        vpxord(zmm_zero, zmm_zero, zmm_zero);
        if (jcp.relu_negative_slope == 0 || jcp.ver == ver_4vnni) {
            zmm_relu_ns = zmm_zero;
        } else {
            mov(imm_addr64, float2int(jcp.relu_negative_slope));
            vmovq(xmm_relu_ns, imm_addr64);
            vbroadcastss(zmm_relu_ns, xmm_relu_ns);
        }
        cmp(reg_channel, jcp.nb_ic - 1);
        jl(store_label, T_NEAR);
        const unsigned char _cmp_lt_os = 1;
        for (int k = 0; k < jcp.nb_oc_blocking; k++)
            for (int j = 0; j < ur_w; j++){
                Opmask kmask = Opmask(7);
                Zmm zmm = zmm_out(j, k);
                vcmp(kmask, zmm, zmm_zero, _cmp_lt_os);
                vmul(zmm, kmask, zmm, zmm_relu_ns);
            }
    }

    L(store_label);
    int saved_cache_line = -1;
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = jcp.is_lastconv
                ? zmm_out(j, k) | KMASK_FLCONV: zmm_out(j, k);
            int aux_output_offset
                = typesize * (k * jcp.oh * jcp.ow + j) * jcp.oc_block;
            vmovups(EVEX_compress_addr(reg_out, aux_output_offset), zmm);
            IF_NEW_CACHE_LINE(saved_cache_line, aux_output_offset,
                              mic_prefetcht0(EVEX_compress_addr(
                                      reg_out_prf, aux_output_offset)));
        }
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_4fma_1st(int ur_w,
        int pad_l, int pad_r)
{

    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    Label kh_label;

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_inp_prf, reg_inp_prf);

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    Label skip_kh_loop;
    if (jcp.kh <= jcp.t_pad) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    L(kh_label);
    for (int ki = 0; ki < kw; ki += 4) {
        for (int ic = 0; ic < ic_block; ic++) {
            for (int i = 0; i < 4; i++) {
                int aux_ker_offset = jcp.typesize_in
                    * ((ki + i) * oc_block + ic * kw * jcp.kh * oc_block);
                if (ki + i < kw)
                    vmovups(zmm_ker(i),
                        EVEX_compress_addr(aux_reg_ker, aux_ker_offset));
                else
                    vpxord(zmm_ker(i), zmm_ker(i), zmm_ker(i));
            }

            int j_start = get_ow_start(ki, pad_l);
            int j_end = get_ow_end(ur_w, ki, pad_r);

            for (int j = j_start, prf_count=0; j < j_end; j++) {
                int aux_input_offset = jcp.typesize_in
                            * ((ki + j * stride_w - pad_l) + ic * iw * ih);
                v4fmaddps(zmm_out(j, 0), zmm_ker(0),
                    EVEX_compress_addr(aux_reg_inp, aux_input_offset));
                if (ki + prf_count < kw && prf_count < 4
                    && ((ki < 2 && j % 4) || j % 2)) {
                    int aux_ker_offset = jcp.typesize_in
                        * ((ki + prf_count) * oc_block
                        + ic * kw * jcp.kh * oc_block + kw * oc_block);
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                        aux_ker_offset));
                    prf_count++;
                }
                if (ki == 0
                    && j % (64 / (stride_w * jcp.typesize_in)) == 0) {
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_inp_prf,
                        aux_input_offset));
                }
                if (ki == 1
                    && j % (64 / (stride_w * jcp.typesize_in)) == 0) {
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_inp,
                        aux_input_offset+jcp.typesize_in * iw));
                }
            }
        }
    }
    add(aux_reg_ker, jcp.typesize_in * kw * oc_block);
    add(aux_reg_inp, jcp.typesize_in * iw);
    add(aux_reg_inp_prf, jcp.typesize_in * iw);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    L(skip_kh_loop);

    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_4fma(int ur_w,
        int pad_l, int pad_r)
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label, last_iter_label, loop_end_label;
    int ker_load_number = 4;
    int shift_kernel_ptr = typesize * jcp.kw * jcp.oc_block * jcp.ic_block;
    int shift_input_ptr = typesize * jcp.iw * jcp.ic_block;

    bool check_last_kh = (jcp.kh > 3);
    bool one_fma_one_pref = (jcp.ow > 28);

    int oi_ipref_t0 = get_ow_start(0, pad_l);
    int oi_ipref_t1 = oi_ipref_t0;
    int ow_end_ipref = get_ow_end(ur_w, 0, pad_r);

    assert(jcp.oc % jcp.nb_oc_blocking == 0);

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_ker_prf, reg_ker_prf);
    mov(aux_reg_inp_prf, reg_inp_prf);

    auto kernel_offset = [=](int ocb, int ic, int ki) {
        int blk_idx = ocb * jcp.nb_ic * jcp.kh * jcp.kw + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int ic_offset = ic * jcp.oc_block;
        return typesize * (blk_offset + ic_offset);
    };
    auto kernel_loads = [=](int ki, int ic, int kk) {
        for (int ii = 0; ii < ker_load_number; ii++) {
            int aux_kernel_offset = kernel_offset(kk, ic + ii, ki);
            vmovups(zmm_ker(ii),
                EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
        }
    };
    auto prefetch_inp_next_kh = [&](int ki, int ki_start, int cnt0, int cnt1) {
        if (cnt1 >= ker_load_number && cnt0 >= ker_load_number
            && ki >= ki_start && oi_ipref_t0 < ow_end_ipref) {
            int aux_inp_offset = typesize * ((oi_ipref_t0
                    * stride_w - pad_l) * ic_block + jcp.iw * ic_block);
            prefetcht0(EVEX_compress_addr(aux_reg_inp,
                    aux_inp_offset));
            oi_ipref_t0++;
        }
    };

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    Label skip_kh_loop;
    if (jcp.kh <= jcp.t_pad) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }

    align(16);
    L(kh_label);
    if (check_last_kh) {
        for (int ki = 0; ki < kw; ki++)
            for (int ic = 0; ic < ic_block; ic += 4)
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    bool last_kernel_loads = (kk == jcp.nb_oc_blocking - 1
                        && ki == kw - 1 && (ic + 4) == ic_block);

                    if (last_kernel_loads) {
                        cmp(reg_kj, 1);
                        je(last_iter_label, T_NEAR);
                    }

                    kernel_loads(ki, ic, kk);
                    for (int oi = get_ow_start(ki, pad_l), prf_count_t1 = 0,
                        prf_count_t0 = 0; oi  < get_ow_end(ur_w, ki, pad_r); oi++) {
                        int aux_input_offset =  typesize
                            * ((ki + oi * stride_w - pad_l) * ic_block + ic);
                        v4fmaddps(zmm_out(oi, kk), zmm_ker(0),
                            EVEX_compress_addr(aux_reg_inp, aux_input_offset));

                        if (oi % 2) {
                            if (prf_count_t0 < 4) {
                                int aux_kernel_prf;
                                if (last_kernel_loads)
                                    aux_kernel_prf= kernel_offset(0,
                                        prf_count_t0 + ic + 4
                                        - ic_block, 0) + typesize * kw
                                        * oc_block * ic_block;
                                else
                                    aux_kernel_prf = kernel_offset(kk, ic + 4
                                        + prf_count_t0, ki);
                                mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                                    aux_kernel_prf));
                                prf_count_t0++;
                            } else if (prf_count_t1 < 4) {
                                mic_prefetcht1(EVEX_compress_addr(
                                    aux_reg_ker_prf, kernel_offset(kk, ic
                                    + prf_count_t1, ki)));
                                prf_count_t1++;
                            }
                        } else
                           prefetch_inp_next_kh(ki, 2, prf_count_t0,
                               prf_count_t1);
                    }

                    if (last_kernel_loads) {
                        jmp(loop_end_label, T_NEAR);

                        L(last_iter_label);

                        kernel_loads(ki, ic, kk);
                        for (int oi = get_ow_start(ki, pad_l), prf_count_t1 = 0,
                             prf_count_t0 = 0; oi  < get_ow_end(ur_w, ki, pad_r);
                             oi++) {
                            int aux_input_offset =  typesize
                               * ((ki + oi * stride_w - pad_l) * ic_block + ic);
                            v4fmaddps(zmm_out(oi, kk), zmm_ker(0),
                                EVEX_compress_addr(aux_reg_inp,
                                    aux_input_offset));
                            if (oi % 2) {
                                if (prf_count_t0 < 4) {
                                    mic_prefetcht0(EVEX_compress_addr(
                                        aux_reg_ker_prf, kernel_offset(0,
                                        prf_count_t0, 0)));
                                    prf_count_t0++;
                                } else if (prf_count_t1 < 4) {
                                    mic_prefetcht1(EVEX_compress_addr(
                                        aux_reg_ker_prf, kernel_offset(kk,
                                        ic + prf_count_t1, ki)));
                                    prf_count_t1++;
                                }
                            }
                        }
                        L(loop_end_label);
                    }
                }
    } else {
        for (int ki = 0; ki < kw; ki++)
            for (int ic = 0; ic < ic_block; ic += 4)
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    kernel_loads(ki, ic, kk);
                    for (int oi = get_ow_start(ki, pad_l), prf_count_t1 = 0,
                        prf_count_t0 = 0;
                        oi  < get_ow_end(ur_w, ki, pad_r); oi++) {
                        int aux_input_offset =  typesize
                            * ((ki + oi * stride_w - pad_l) * ic_block + ic);
                        v4fmaddps(zmm_out(oi, kk), zmm_ker(0),
                            EVEX_compress_addr(aux_reg_inp, aux_input_offset));

                        if (one_fma_one_pref) {
                            if (prf_count_t0 < 4) {
                                int aux_kernel_prf;
                                if (kk == jcp.nb_oc_blocking - 1 && ki == kw - 1
                                    && ic + 4 >= ic_block)
                                    aux_kernel_prf = kernel_offset(0,
                                        prf_count_t0 + ic + 4 - ic_block, 0)
                                        + typesize * kw * oc_block * ic_block;
                                else
                                    aux_kernel_prf = kernel_offset(kk,
                                        ic + 4 + prf_count_t0, ki);
                                mic_prefetcht0(EVEX_compress_addr(
                                    aux_reg_ker, aux_kernel_prf));
                                prf_count_t0++;
                            } else if (prf_count_t1 < 4) {
                                mic_prefetcht1(EVEX_compress_addr(
                                    aux_reg_ker_prf, kernel_offset(kk,
                                    ic + prf_count_t1, ki)));
                                prf_count_t1++;
                            } else
                                prefetch_inp_next_kh(ki, 0, prf_count_t0,
                                    prf_count_t1);
                        } else {
                            if ((oi % 2) && (prf_count_t0 < 4)) {
                                mic_prefetcht0(EVEX_compress_addr(
                                    aux_reg_ker_prf, kernel_offset(kk,
                                    ic + prf_count_t0, ki)));
                                prf_count_t0++;
                            }
                            if (!(oi % 2) && ki == 0 && ic == 0 && kk == 0)
                                mic_prefetcht1(EVEX_compress_addr(
                                    aux_reg_inp_prf, aux_input_offset));
                            if (!(oi % 2) && ki == 1 && ic == 0 && kk == 0)
                                mic_prefetcht0(EVEX_compress_addr(aux_reg_inp,
                                  aux_input_offset + typesize * iw * ic_block));
                        }
                    }
                }
    }

    add(aux_reg_ker, shift_kernel_ptr);
    add(aux_reg_inp, shift_input_ptr);
    add(aux_reg_ker_prf, shift_kernel_ptr);
    add(aux_reg_inp_prf, shift_input_ptr);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    L(skip_kh_loop);

    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_fma(int ur_w, int pad_l,
        int pad_r)
{
    bool prf_ker = true;
    bool prf_inp = true;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label;

    int ker_pipeline_depth = nstl::min(4, jcp.ic_block);
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(ic_block >= ker_pipeline_depth);

    int num_ker_loads = ic_block * nb_oc_block * kw;
    const int simd_w = 16;
    int num_ker_prfs = prf_ker ? num_ker_loads : 0;
    int num_inp_prfs = prf_inp ?
            ur_w * nstl::min(kw, stride_w) + nstl::max(0, kw - stride_w) :
            0;
    if (jcp.is_1stconv && prf_inp) {
        num_inp_prfs = div_up(num_inp_prfs, simd_w) * ic_block;
    }
    int num_prfs = num_ker_prfs + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w;
    int prf_inst_spacing
            = (prf_ker || prf_inp) ? nstl::max(1, num_fmas / num_prfs) : 1;
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);

    prepare_output(ur_w);

    mov(aux_reg_inp_prf, reg_inp_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);
    mov(reg_kj, reg_kh);
    Label skip_kh_loop;
    if (jcp.kh <= jcp.t_pad) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    align(16);
    L(kh_label);
    {
        int step = 0;
        int ker_prfs = 0;
        int fma_idx = 0;
        int saved_kcache_line = -1;

        for (int ki = 0; ki < kw; ki++) {
            for (int ic = 0; ic < ic_block; ic++) {
                int aux_kernel_offset = 0;
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        aux_kernel_offset = get_kernel_offset(ki, ic, 0, i);
                        Zmm zmm = jcp.is_lastconv
                            ? zmm_ker(i) | KMASK_FLCONV : zmm_ker(i);
                        vmovups(zmm, EVEX_compress_addr(
                                        aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                        = (step + load_offset) % ker_pipeline_depth;
                    aux_kernel_offset = get_kernel_offset(ki,ic,0,load_offset);
                    Zmm zmm = jcp.is_lastconv
                        ? zmm_ker(ker_load_reg_idx) | KMASK_FLCONV
                        : zmm_ker(ker_load_reg_idx);
                    vmovups(zmm, EVEX_compress_addr(aux_reg_ker,
                                                    aux_kernel_offset));
                }

                bool ker_prf_inserted = false;
                Zmm zmm_kernel = zmm_ker(step % ker_pipeline_depth);
                int j_start = get_ow_start(ki, pad_l);
                int j_end = get_ow_end(ur_w, ki, pad_r);
                for (int j = j_start; j < j_end; j++) {
                    int aux_input_offset = get_input_offset(ki, ic, j, pad_l);
                    Zmm zmm_dst = jcp.is_lastconv
                        ? zmm_out(j, 0) | KMASK_FLCONV: zmm_out(j, 0);
                    vfmadd231ps(zmm_dst, zmm_kernel,
                       EVEX_compress_addr(aux_reg_inp, aux_input_offset, true));
                    fma_idx++;

                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (prf_ker && !ker_prf_inserted
                                && ker_prfs < num_ker_prfs) {
                            int ker_prf_offset
                                    = jcp.typesize_in * ker_prfs * jcp.oc_block;
                            IF_NEW_CACHE_LINE(saved_kcache_line, ker_prf_offset,
                                    mic_prefetcht2(EVEX_compress_addr(
                                    aux_reg_ker_prf, ker_prf_offset)));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else if (prf_inp) {
                            int inp_prf_idx = prf_slot_idx - ker_prfs;
                            if (inp_prf_idx < num_inp_prfs) {
                                int inp_prf_stride = nstl::max(kw, stride_w);
                                int inp_prf_offset;
                                if (!jcp.is_1stconv) {
                                    inp_prf_offset
                                            = ic_block * jcp.typesize_in
                                            * ((inp_prf_idx / kw)
                                            * inp_prf_stride
                                            + (inp_prf_idx % kw));
                                } else {
                                    int ic_prf_stride = jcp.typesize_in*iw*ih;
                                    int iw_prf_stride = jcp.typesize_in*simd_w;
                                    inp_prf_offset = ((inp_prf_idx / ic_block)
                                            * iw_prf_stride
                                            + (inp_prf_idx % ic_block)
                                            * ic_prf_stride);
                                }
                                mic_prefetcht0(EVEX_compress_addr(
                                        aux_reg_inp_prf, inp_prf_offset));
                            }
                        }
                    }
                }
                step++;
            }
        }
        add(aux_reg_ker, jcp.typesize_in * kw * oc_block * ic_block);
        if (prf_ker)
            add(aux_reg_ker_prf, jcp.typesize_in * kw * oc_block * ic_block);
        int inp_mul = !jcp.is_1stconv ? ic_block : 1;
        add(aux_reg_inp, jcp.typesize_in * iw * inp_mul);
        if (prf_inp)
            add(aux_reg_inp_prf, jcp.typesize_in * iw * inp_mul);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_4vnni(
        int ur_w, int pad_l, int pad_r)
{
    Label kh_label;
    const int ker_reg_base_idx = 28;
    const int ker_load_number = 4;
    const int shift_kernel_ptr = jcp.typesize_in * jcp.kw
                               * jcp.oc_block * jcp.ic_block;
    const int shift_input_ptr = jcp.typesize_in * jcp.iw * jcp.ic_block;

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_ker_prf, reg_ker_prf);
    mov(aux_reg_inp_prf, reg_inp_prf);

    prepare_output(ur_w);

    Label skip_kh_loop;
    mov(reg_kj, reg_kh);
    if (jcp.kh <= jcp.t_pad) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    L(kh_label); {
        for (int ki = 0; ki < jcp.kw; ki++) {
            for (int ic = 0; ic < jcp.ic_block / 2; ic += 4) {
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    for (int ii = 0; ii < ker_load_number; ii++) {
                        int kernel_offset = get_kernel_offset(ki, ic, kk, ii);
                        vmovups(Zmm(ker_reg_base_idx+ii),
                            EVEX_compress_addr(aux_reg_ker, kernel_offset));
                    }
                    int ow_start = get_ow_start(ki, pad_l);
                    int ow_end = get_ow_end(ur_w, ki, pad_r);
                    for (int oi = ow_start, prf_count = 0; oi  < ow_end; oi++) {
                        int input_offset = get_input_offset(ki,ic,oi,pad_l);
                        vp4dpwssd(Zmm(ur_w*kk + oi), Zmm(ker_reg_base_idx),
                            EVEX_compress_addr(aux_reg_inp, input_offset));
                        if ((oi % 2) && (prf_count < ker_load_number)) {
                            int kernel_offset = get_kernel_offset(
                                ki, ic, kk, prf_count++);
                            prefetcht0(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset));
                        }
                        if (!(oi % 2) && ki == 0 && ic==0 && kk==0) {
                            prefetcht1(EVEX_compress_addr(aux_reg_inp_prf,
                                input_offset));
                        }
                        if (!(oi % 2) && ki == 1 && ic==0 && kk==0) {
                            prefetcht0(EVEX_compress_addr(aux_reg_inp,
                                input_offset + shift_input_ptr));
                        }
                    }
                }
            }
        }
        add(aux_reg_ker_prf, shift_kernel_ptr);
        add(aux_reg_inp_prf, shift_input_ptr);
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);
    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop(int ur_w,
        int pad_l, int pad_r)
{
    if (jcp.ver == ver_4vnni)
        compute_loop_4vnni(ur_w, pad_l, pad_r);
    else if (jcp.ver == ver_4fma)
        if(jcp.is_1stconv)
            compute_loop_4fma_1st(ur_w, pad_l, pad_r);
        else
            compute_loop_4fma(ur_w, pad_l, pad_r);
    else if (jcp.ver == ver_fma)
        compute_loop_fma(ur_w, pad_l, pad_r);
    else
        assert(!"unknown convolution version");
}

void jit_avx512_common_conv_fwd_kernel::generate()
{
    int iw = jcp.iw;
    int ow = jcp.ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int inp_mult = !jcp.is_1stconv ? ic_block : 1;
    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = jcp.typesize_in * (ur_w * stride_w * inp_mult);
    int out_shift = jcp.typesize_out * (ur_w * oc_block);

    preamble();

    if (jcp.is_lastconv && (jcp.ver == ver_fma))
        SET_FLCONV_KMASK(jcp.oc_block);

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    int r_pad = nstl::max(0, (ow - 1) * stride_w + (kw - 1) - (iw + l_pad - 1));
    if (ow == ur_w) {
        mov(reg_inp_prf, ptr[param1 + GET_OFF(src_prf)]);
        mov(reg_out_prf, ptr[param1 + GET_OFF(dst_prf)]);
        compute_loop(ur_w, l_pad, r_pad);
    } else {
        mov(reg_inp_prf, reg_inp);
        mov(reg_out_prf, reg_out);
        int n_oi = ow / ur_w;

        int r_pad1 = (ur_w * n_oi - 1) * stride_w + kw - 1 - (iw + l_pad - 1);
        xor_(reg_oi, reg_oi);
        if (l_pad > 0) {
            add(reg_inp_prf, inp_shift_pad);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, l_pad, 0);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            inc(reg_oi);

            if (r_pad1 > 0)
                n_oi--;
        }
        if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
            if (l_pad <= 0 && r_pad1 > 0)
                n_oi--;
            Label ow_loop_label;
            L(ow_loop_label);
            {
                add(reg_inp_prf, inp_shift);
                add(reg_out_prf, out_shift);
                compute_loop(ur_w, 0, 0);
                add(reg_inp, inp_shift);
                add(reg_out, out_shift);
                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_pad1 > 0) {
            add(reg_inp_prf, inp_shift);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, 0, r_pad1);
            add(reg_inp, inp_shift);
            add(reg_out, out_shift);
        }
        if (ur_w_tail != 0) {
            add(reg_inp_prf, inp_shift);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w_tail, 0, r_pad);
        }
    }

    postamble();
}

status_t jit_avx512_common_conv_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd, bool with_relu, double relu_negative_slope)
{
    using namespace prop_kind;

    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    jcp = zero<decltype(jcp)>();
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.src_fmt = src_d.format();
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;
    jcp.ur_h = 1;
    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.oc_block = (jcp.oc % simd_w != 0) ? jcp.oc : simd_w;

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    if (jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    jcp.is_1stconv = jcp.ic % simd_w;
    if (jcp.is_1stconv) {
        if (!one_of(jcp.ic, 1, 3, 4))
            return status::unimplemented;
        if (src_d.format() == any)
            CHECK(src_pd.set_format(nchw));
        if (src_d.format() != nchw)
            return status::unimplemented;
    } else {
        if (src_d.format() == any)
            CHECK(src_pd.set_format(nChw16c));
        if (src_d.format() != nChw16c)
            return status::unimplemented;
    }

    jcp.is_lastconv = jcp.oc % simd_w;
    if (jcp.is_lastconv) {
        if (!one_of(jcp.oc, 3, 4)) // OC=1 may have performance issue
            return status::unimplemented;
        if (jcp.is_1stconv)
            return status::unimplemented;
        if (dst_d.format() == any)
            CHECK(dst_pd.set_format(nhwc));
        if (dst_d.format() != nhwc)
            return status::unimplemented;
        if (with_relu || with_groups)
            return status::unimplemented;
    } else {
        if (dst_d.format() == any)
            CHECK(dst_pd.set_format(nChw16c));
        if (dst_d.format() != nChw16c)
            return status::unimplemented;
    }

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }

    if (mayiuse(avx512_mic_4ops) &&
            src_d.data_type() == data_type::s16
         && weights_d.data_type() == data_type::s16
         && dst_d.data_type() == data_type::s32)
    {
        if (jcp.is_1stconv || jcp.is_lastconv)
            return status::unimplemented;

        jcp.ver = ver_4vnni;
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);

        const auto w_format = with_groups ? gOIhw8i16o2i : OIhw8i16o2i;
        if (weights_d.format() == any)
            CHECK(weights_pd.set_format(w_format));
        if (!one_of(weights_d.format(), gOIhw8i16o2i, OIhw8i16o2i))
            return status::unimplemented;
    } else if (mayiuse(avx512_common) &&
            src_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && dst_d.data_type() == data_type::f32) {
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops) && !jcp.is_lastconv)
           jcp.ver = ver_4fma;

        if (jcp.is_1stconv) {
            // TODO: fix & remove constraints below
            if (jcp.l_pad != 0 || jcp.r_pad != 0
                || jcp.b_pad != 0 || jcp.t_pad != 0
                || (jcp.kw < 7 && jcp.kh < 7))
                jcp.ver = ver_fma;
            if (jcp.ver == ver_4fma) {
                const auto w_format = (with_groups) ? gOihw16o : Oihw16o;
                if (weights_d.format() == any)
                    CHECK(weights_pd.set_format(w_format));
                if (!one_of(weights_d.format(), Oihw16o, gOihw16o))
                    return status::unimplemented;
            } else {
                const auto w_format = (with_groups) ? gOhwi16o : Ohwi16o;
                if (weights_d.format() == any)
                    CHECK(weights_pd.set_format(w_format));
                if (!one_of(weights_d.format(), Ohwi16o, gOhwi16o))
                    return status::unimplemented;
            }
        } else if (jcp.is_lastconv) {
            if (weights_d.format() == any)
                CHECK(weights_pd.set_format(Ihw16io));
            if (weights_d.format() != Ihw16io)
                return status::unimplemented;
        } else {
            const auto w_format = (with_groups) ? gOIhw16i16o : OIhw16i16o;
            if (weights_d.format() == any)
                CHECK(weights_pd.set_format(w_format));
            if (!one_of(weights_d.format(), OIhw16i16o, gOIhw16i16o))
                return status::unimplemented;
        }
    } else {
        return status::unimplemented;
    }

    if (jcp.is_1stconv) {
        jcp.ur_w = nstl::min(jcp.ow, regs);
    } else {
        for (int ur_w = regs; ur_w > 0; --ur_w) {
            if (jcp.ow % ur_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
        }
        if (jcp.ur_w == 1) {
            jcp.ur_w = nstl::min(jcp.ow, regs);
        }
    }
    // TODO (Tanya): currenly applied to Segnet convolutions only.
    // Need to try for other topologies
    if (jcp.ow > 150 && jcp.ur_w < regs/2)
        jcp.ur_w = regs;

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = (jcp.ur_w * n_oi - 1) * jcp.stride_w + jcp.kw - jcp.iw
            - jcp.l_pad;
    if (jcp.l_pad > 0 && r_pad > 0)
        n_oi--;

    bool large_code_size = jcp.ur_w != jcp.ow && jcp.l_pad > 0 && r_pad > 0
            && ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1));
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.ic_block * jcp.kw;
        int mult = 1;
        if (jcp.l_pad > 0) mult += 1;
        if (r_pad > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if (ur_w * mult * num_ops_per_reg * 9.0 < max_code_size) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    bool args_ok = true
        && jcp.l_pad <= jcp.ur_w
        && implication(!jcp.is_1stconv, jcp.ic % simd_w == 0);
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + jcp.kw - jcp.iw - jcp.l_pad);
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    if (one_of(jcp.ver, ver_4vnni, ver_4fma) && !jcp.is_1stconv)
        for (int i = jcp.nb_oc; i > 0; i--)
            if (i * jcp.ur_w <= regs && jcp.nb_oc % i == 0) {
                jcp.nb_oc_blocking = i;
                break;
            }

    pick_loop_order(jcp);

    return status::success;
}

void jit_avx512_common_conv_bwd_data_kernel_f32::prepare_output(int ur_w)
{
    int saved_cache_line = -1;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j  < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
            int aux_src_offset
                = typesize * (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            IF_NEW_CACHE_LINE(saved_cache_line, aux_src_offset, prefetcht1(
                    EVEX_compress_addr(reg_src_prf, aux_src_offset)));
        }
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::store_output(int ur_w)
{
    Label no_update_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    je(no_update_label, T_NEAR);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = jcp.is_1stconv ?
                zmm_out(j, k) | KMASK_FLCONV : zmm_out(j, k);
            int aux_src_offset
                = typesize * (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            vadd(zmm, reg_src, aux_src_offset);
        }
    }

    L(no_update_label);
    int saved_cache_line = -1;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = jcp.is_1stconv ?
                zmm_out(j, k) | KMASK_FLCONV : zmm_out(j, k);
            int aux_src_offset
                = typesize * (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            vmovups(EVEX_compress_addr(reg_src, aux_src_offset), zmm);
            IF_NEW_CACHE_LINE(saved_cache_line, aux_src_offset, mic_prefetcht0(
                    EVEX_compress_addr(reg_src_prf, aux_src_offset)));
        }
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_4fma(int ur_w,
        int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label, last_iter_label, loop_end_label;
    int ker_load_number = 4;
    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int shift_dst_ptr = typesize * ow * oc_block;
    int ii_dpref_t0 = get_iw_start(0, l_overflow);
    int ii_dpref_t1 = ii_dpref_t0;
    int iw_end_ipref = get_iw_end(ur_w, 0, r_overflow);

    bool check_last_kh = (jcp.kh > 3 || (jcp.kh > 1 && jcp.iw < 112));
    bool one_fma_one_pref = (jcp.iw > 224);

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_dst_prf, reg_dst_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);

    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };
    auto kernel_loads = [=](int ki, int oc, int kk) {
        for (int ii = 0; ii < ker_load_number; ii++) {
            int aux_kernel_offset = kernel_offset(kk, oc + ii, ki);
            vmovups(zmm_ker(ii),
                EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
        }
    };
    auto prefetch_dst_next_kh = [&](int ki, int ki_start, int cnt0, int cnt1) {
        if (cnt1 >= ker_load_number && cnt0 >= ker_load_number
            && ki >= ki_start && ii_dpref_t0 < iw_end_ipref) {
            int aux_dst_offset = typesize * ((ii_dpref_t0
                + jcp.l_pad) * oc_block + jcp.ow * oc_block);
            prefetcht0(EVEX_compress_addr(aux_reg_dst, aux_dst_offset));
            ii_dpref_t0++;
        }
    };

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    align(16);
    L(kh_label);
    if (check_last_kh) {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            bool last_kernel_loads = (kk == jcp.nb_ic_blocking - 1
                && ki == kw - 1 && (oc + 4) == oc_block);

            if (last_kernel_loads) {
                cmp(reg_kj, 1);
                je(last_iter_label, T_NEAR);
            }

            kernel_loads(ki, oc, kk);
            for (int ii = get_iw_start(ki, l_overflow),
                    prf_count_t0 = 0, prf_count_t1 = 0;
                    ii < get_iw_end(ur_w, ki, r_overflow); ii++) {
                int aux_dst_offset = typesize
                    * ((ii + jcp.l_pad - ki) * oc_block + oc);
                v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                    EVEX_compress_addr(aux_reg_dst, aux_dst_offset));

                if (ii % 2) {
                    if (prf_count_t0 < 4) {
                        int aux_kernel_prf;
                        if (last_kernel_loads)
                            aux_kernel_prf= kernel_offset(0, prf_count_t0
                                + oc + 4 - oc_block, 0) + typesize * kw
                                * oc_block * ic_block;
                        else
                            aux_kernel_prf = kernel_offset(kk, oc + 4
                                + prf_count_t0, ki);
                        mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                            aux_kernel_prf));
                        prf_count_t0++;
                    } else if (prf_count_t1 < 4) {
                        mic_prefetcht1(EVEX_compress_addr(aux_reg_ker_prf,
                            kernel_offset(kk, oc + prf_count_t1, ki)));
                        prf_count_t1++;
                    }
                } else
                    prefetch_dst_next_kh(ki, 2, prf_count_t0, prf_count_t1);
            }
            if (last_kernel_loads) {
                jmp(loop_end_label, T_NEAR);

                L(last_iter_label);

                kernel_loads(ki, oc, kk);
                for (int ii = get_iw_start(ki, l_overflow),
                        prf_count_t0 = 0, prf_count_t1 = 0;
                        ii < get_iw_end(ur_w, ki, r_overflow); ii++) {
                    int aux_dst_offset = typesize
                        * ((ii + jcp.l_pad - ki) * oc_block + oc);
                    v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                            EVEX_compress_addr(aux_reg_dst, aux_dst_offset));
                    if (ii % 2) {
                        if (prf_count_t0 < 4) {
                            mic_prefetcht0(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset(0, prf_count_t0, 0)));
                            prf_count_t0++;
                        } else if (prf_count_t1 < 4) {
                            mic_prefetcht1(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset(kk, oc + prf_count_t1, ki)));
                            prf_count_t1++;
                        }
                    }
                }
                L(loop_end_label);
            }
        }
    } else {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            kernel_loads(ki, oc, kk);

            for (int ii = get_iw_start(ki, l_overflow),
                    prf_count_t0 = 0, prf_count_t1 = 0;
                    ii  < get_iw_end(ur_w, ki, r_overflow); ii++) {
                int aux_dst_offset = typesize
                    * ((ii + jcp.l_pad - ki) * oc_block + oc);
                v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                    EVEX_compress_addr(aux_reg_dst, aux_dst_offset));

                if (one_fma_one_pref) {
                    if (prf_count_t0 < 4) {
                        int aux_kernel_prf;
                        if (kk == jcp.nb_ic_blocking - 1 && ki == kw - 1
                            && oc + 4 >= oc_block)
                            aux_kernel_prf = kernel_offset(0,
                                prf_count_t0 + oc + 4 - oc_block, 0)
                                + typesize * kw * oc_block * ic_block;
                        else
                            aux_kernel_prf = kernel_offset(kk,
                                oc + 4 + prf_count_t0, ki);
                        mic_prefetcht0(EVEX_compress_addr(
                            aux_reg_ker, aux_kernel_prf));
                        prf_count_t0++;
                    } else if (prf_count_t1 < 4) {
                        mic_prefetcht1(EVEX_compress_addr(aux_reg_ker_prf,
                            kernel_offset(kk, oc + prf_count_t1, ki)));
                        prf_count_t1++;
                    } else
                        prefetch_dst_next_kh(ki, 0, prf_count_t0,
                            prf_count_t1);
                } else {
                    if ((ii % 2) && (prf_count_t1 < 4)) {
                        int aux_kernel_prf = kernel_offset(kk,
                            oc + prf_count_t1, ki);
                        mic_prefetcht1(EVEX_compress_addr(
                            aux_reg_ker_prf, aux_kernel_prf));
                        prf_count_t1++;
                    }
                    if (!(ii % 2) && ki == 0 && oc == 0 && kk == 0)
                        mic_prefetcht1(EVEX_compress_addr(aux_reg_dst_prf,
                            aux_dst_offset));
                    if (!(ii % 2) && ki == 1 && oc == 0 && kk == 0)
                        mic_prefetcht0(EVEX_compress_addr(aux_reg_dst,
                            aux_dst_offset + shift_dst_ptr));
                }
            }
        }
    }

    add(aux_reg_ker, shift_ker_ptr);
    sub(aux_reg_dst, shift_dst_ptr);
    add(aux_reg_ker_prf, shift_ker_ptr);
    sub(aux_reg_dst_prf, shift_dst_ptr);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    store_output(ur_w);
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_4vnni(int ur_w,
        int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label;

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_dst_prf, reg_dst_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);

    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return jcp.typesize_in * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    L(kh_label); {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block / 2; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            for (int ii = 0; ii < 4; ii++) {
                int aux_kernel_offset = kernel_offset(kk, 2 * (oc + ii), ki);
                vmovups(zmm_ker(ii),
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
            }

            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int jj = jj_start, prf_count = 0; jj  < jj_end; jj++) {
                int aux_dst_offset = jcp.typesize_in
                    * ((jj + jcp.l_pad - ki) * oc_block + 2 * oc);
                vp4dpwssd(zmm_out(jj, kk), zmm_ker(0),
                        EVEX_compress_addr(aux_reg_dst, aux_dst_offset));

                if ((jj % 2) && (prf_count < 4)) {
                    int aux_kernel_prf = kernel_offset(kk, oc + prf_count, ki);
                    mic_prefetcht1(EVEX_compress_addr(
                        aux_reg_ker_prf, aux_kernel_prf));
                    prf_count++;
                }
                if (!(jj % 2) && ki == 0 && oc == 0 && kk == 0) {
                    mic_prefetcht1(EVEX_compress_addr(aux_reg_dst_prf,
                        aux_dst_offset));
                }
                if (!(jj % 2) && ki == 1 && oc == 0 && kk == 0) {
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_dst,
                        aux_dst_offset + jcp.typesize_in * ow * oc_block));
                }
            }
        }

        add(aux_reg_ker, jcp.typesize_in * kw * oc_block * ic_block);
        sub(aux_reg_dst, jcp.typesize_in * ow * oc_block);
        add(aux_reg_ker_prf, jcp.typesize_in * kw * oc_block * ic_block);
        sub(aux_reg_dst_prf, jcp.typesize_in * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    store_output(ur_w);
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma(int ur_w,
        int l_overflow, int r_overflow)
{
    Label kh_label;
    Label store_output_label;
    int kw    = jcp.kw;
    int ow    = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad    = jcp.l_pad;
    int stride_w = jcp.stride_w;

    int ker_pipeline_depth = nstl::min(4, oc_block);
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;
    int num_inp_prfs = ur_w * nstl::min(kw, stride_w)
                       + nstl::max(0, kw - stride_w);
    int num_prfs = num_ker_loads + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w / stride_w;
    int prf_inst_spacing = nstl::max(1, num_fmas / num_prfs);
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    prepare_output(ur_w);

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);

    mov(aux_reg_dst_prf, reg_dst_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);

    mov(reg_kj, reg_kh);
    cmp(reg_kj, 0);
    je(store_output_label, T_NEAR);
    L(kh_label); {
        int step = 0;
        int ker_prfs = 0;
        int fma_idx = 0;
        int saved_kcache_line = -1;
        int saved_dcache_line = -1;

        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        int aux_kernel_offset = typesize * ((oc + i) * ic_block
                                + ki * ic_block * oc_block);
                        Zmm zmm = jcp.is_1stconv
                            ? zmm_ker(i) | KMASK_FLCONV : zmm_ker(i);
                        vmovups(zmm, EVEX_compress_addr(aux_reg_ker,
                                                        aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                        = (step + load_offset) % ker_pipeline_depth;
                    int aux_kernel_offset = typesize * ((oc + load_offset)
                            * ic_block + ki * ic_block * oc_block);
                    Zmm zmm = jcp.is_1stconv
                        ? zmm_ker(ker_load_reg_idx) | KMASK_FLCONV
                        : zmm_ker(ker_load_reg_idx);
                    vmovups(zmm, EVEX_compress_addr(aux_reg_ker,
                                                    aux_kernel_offset));
                }

                bool ker_prf_inserted = false;
                auto zmm_kernel = zmm_ker(step % ker_pipeline_depth);

                int jj_start = get_iw_start(ki, l_overflow);
                int jj_end   = get_iw_end(ur_w, ki, r_overflow);
                assert(stride_w != 1 ||
                        jj_start == nstl::max(0, l_overflow - (kw - 1) + ki));
                assert(stride_w != 1 ||
                        jj_end == ur_w - nstl::max(0, r_overflow - ki));

                for (int jj = jj_start; jj  < jj_end; jj += stride_w) {
                    assert((jj + l_pad - ki) % stride_w == 0);
                    int aux_dst_offset = typesize * (((jj + l_pad - ki)
                                / stride_w) * jcp.oc_block + oc);
                    Zmm zmm_src = jcp.is_1stconv ?
                        zmm_out(jj, 0) | KMASK_FLCONV : zmm_out(jj, 0);
                    vfmadd231ps(zmm_src, zmm_kernel,
                        EVEX_compress_addr(aux_reg_dst, aux_dst_offset, true));
                    fma_idx++;

                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (!ker_prf_inserted && ker_prfs < num_ker_loads) {
                            int ker_prf_offset = typesize
                                * ker_prfs * jcp.ic_block;
                            IF_NEW_CACHE_LINE(saved_kcache_line, ker_prf_offset,
                                             mic_prefetcht1(EVEX_compress_addr(
                                             aux_reg_ker_prf, ker_prf_offset)));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else {
                            int out_prf_idx = prf_slot_idx - ker_prfs;
                            if (out_prf_idx < num_inp_prfs) {
                                int out_prf_offset
                                    = oc_block * typesize
                                    * ((out_prf_idx / kw) * kw
                                            + (out_prf_idx % kw));
                                IF_NEW_CACHE_LINE(saved_dcache_line,
                                             out_prf_offset, mic_prefetcht0(
                                             EVEX_compress_addr(aux_reg_dst_prf,
                                             out_prf_offset)));
                            }
                        }
                    }
                }
                step++;
            }
        }

        add(aux_reg_ker, typesize * stride_w * kw * oc_block * ic_block);
        sub(aux_reg_dst, typesize * ow * oc_block);
        add(aux_reg_ker_prf, typesize * stride_w * kw * oc_block * ic_block);
        sub(aux_reg_dst_prf, typesize * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(store_output_label); {
        store_output(ur_w);
    }
}

inline void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop(int ur_w,
        int l_overflow, int r_overflow)
{
    if (jcp.ver == ver_4vnni)
        compute_loop_4vnni(ur_w, l_overflow, r_overflow);
    else if (jcp.ver == ver_4fma)
        compute_loop_4fma(ur_w, l_overflow, r_overflow);
    else if (jcp.ver == ver_fma)
        compute_loop_fma(ur_w, l_overflow, r_overflow);
    else
        assert("!unknown convolution version");
}

void jit_avx512_common_conv_bwd_data_kernel_f32::generate()
{
    int iw    = jcp.iw;
    int ow    = jcp.ow;
    int kw    = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w      = jcp.ur_w;
    int ic_block  = jcp.ic_block;
    int oc_block  = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w  = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w) * oc_block;
    int src_shift = jcp.typesize_out * ur_w * ic_block;

    preamble();

    if (jcp.is_1stconv && (jcp.ver == ver_fma))
        SET_FLCONV_KMASK(jcp.ic_block);

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_src_prf, ptr[param + GET_OFF(src_prf)]);
    mov(reg_dst_prf, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    int l_overflow = nstl::max(0, ((kw - 1) - l_pad) / stride_w);
    int r_pad      = nstl::max(0, (stride_w * (ow - 1) + kw - iw - l_pad));
    int r_overflow = nstl::max(0, ((kw - 1) - r_pad) / stride_w);
    int n_oi = iw / ur_w;
    int r_overflow1 = nstl::max(0, ((kw - 1) - (iw - ur_w * n_oi)
                - r_pad) / stride_w);

    if (r_overflow1 > 0) n_oi--;

    if (ur_w == iw) {
        compute_loop(ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(ur_w, l_overflow, r_overflow1);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        add(reg_src_prf, src_shift);
        add(reg_dst_prf, dst_shift);
        if (ur_w_tail != 0)
            compute_loop(ur_w_tail, 0, r_overflow);
    } else {
        xor_(reg_oi, reg_oi);
        if (l_overflow > 0) {
            compute_loop(ur_w, l_overflow, 0);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            add(reg_src_prf, src_shift);
            add(reg_dst_prf, dst_shift);

            inc(reg_oi);
        }
        if ((l_overflow <= 0 && n_oi > 0)
            || (l_overflow >  0 && n_oi > 1)) {
            Label ow_loop_label;
            L(ow_loop_label); {
                compute_loop(ur_w, 0, 0);
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
                add(reg_src_prf, src_shift);
                add(reg_dst_prf, dst_shift);

                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            compute_loop(ur_w, 0, r_overflow1);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            add(reg_src_prf, src_shift);
            add(reg_dst_prf, dst_shift);
        }
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_overflow);
        }
    }

    postamble();
}

status_t jit_avx512_common_conv_bwd_data_kernel_f32::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(avx512_common)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.ih = diff_src_d.dims()[2];
    jcp.iw = diff_src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    if (jcp.stride_w != jcp.stride_h)
        return status::unimplemented;

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    if (jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw
                          - jcp.l_pad);
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih
                          - jcp.t_pad);
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

    const int simd_w = 16;

    jcp.is_1stconv = jcp.ic % simd_w;
    if (jcp.is_1stconv) {
        if (one_of(jcp.ic, 3, 4)) { // IC=1 may have performance issue
            if (with_groups)
                return status::unimplemented;
            if (diff_src_d.format() != nhwc)
                return status::unimplemented;
        } else
            return status::unimplemented;
    } else {
        if (diff_src_d.format() != nChw16c)
            return status::unimplemented;
    }

    jcp.ic_block = (jcp.ic % simd_w) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.is_lastconv = jcp.oc % simd_w;
    if (jcp.is_lastconv) {
        if (one_of(jcp.oc, 1, 3, 4)) {
            if (with_groups)
                return status::unimplemented;
            if (diff_dst_d.format() != nhwc)
                return status::unimplemented;
            if (jcp.is_1stconv)
                return status::unimplemented;
        } else
            return status::unimplemented;
    } else {
        if (diff_dst_d.format() != nChw16c)
            return status::unimplemented;
    }

    jcp.oc_block = (jcp.oc % simd_w) ? jcp.oc : simd_w;
    if (jcp.oc % jcp.oc_block)
        return status::unimplemented;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_h = jcp.stride_h;
    jcp.ur_w = jcp.stride_w;

    int regs = 28;
    if (jcp.iw <= regs)
        jcp.ur_w = jcp.iw;
    else {
        for (int ur_w = regs; ur_w > 0; --ur_w)
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
    }
    int n_oi  = (jcp.iw / jcp.ur_w);
    int l_overflow  = nstl::max(0, ((jcp.kw - 1) - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0, ((jcp.kw - 1) - (jcp.iw - jcp.ur_w * n_oi)
                                - jcp.r_pad) / jcp.stride_w);
    if (r_overflow1 > 0) n_oi--;

    if (mayiuse(avx512_mic_4ops) && !jcp.is_1stconv && !jcp.is_lastconv
           && jcp.stride_w == 1 && jcp.stride_h == 1
           && diff_dst_d.data_type() == data_type::s16
           && weights_d.data_type() == data_type::s16
           && diff_src_d.data_type() == data_type::s32) {
        if (weights_d.format() != (with_groups ? gOIhw8o16i2o : OIhw8o16i2o))
            return status::unimplemented;
        jcp.ver = ver_4vnni;
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);
    } else if (mayiuse(avx512_common) &&
            diff_dst_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && diff_src_d.data_type() == data_type::f32) {
        if (jcp.is_1stconv) {
            if (weights_d.format() != Ohw16oi)
                return status::unimplemented;
        } else if (jcp.is_lastconv) {
            if (weights_d.format() != Ihwo16i)
                return status::unimplemented;
        } else if (weights_d.format() != (with_groups
                                          ? gOIhw16o16i : OIhw16o16i))
            return status::unimplemented;
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops) && !jcp.is_1stconv && !jcp.is_lastconv
            && jcp.stride_w == 1 && jcp.stride_h == 1) {
                jcp.ver = ver_4fma;
            }
    } else {
        return status::unimplemented;
    }

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    if (jcp.ver == ver_4fma)
        for (int i = jcp.nb_ic; i > 0; i--)
            if (i * jcp.ur_w <= regs && jcp.nb_ic % i == 0) {
                jcp.nb_ic_blocking = i;
                break;
            }

    jcp.loop_order = loop_gnc;

    bool large_code_size = (jcp.ur_w != jcp.iw)
         && ((l_overflow <= 0 && n_oi > 0) ||(l_overflow >  0 && n_oi > 1))
         && (r_overflow1 > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow1 > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if ((ur_w / jcp.stride_w) * mult * num_ops_per_reg * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0, (jcp.kw - 1 - jcp.ur_w_tail
                - jcp.r_pad) / jcp.stride_w);
    if (r_overflow_no_tail * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    if ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
        return status::unimplemented;

    pick_loop_order(jcp);

    return status::success;
}

const int jit_avx512_common_conv_bwd_weights_kernel_f32::max_ur_w = 28;

void jit_avx512_common_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers()
{
    Label kh_comeback_label;

    mov(kj, reg_kh);
    L(kh_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = jcp.ver == ver_4fma ? jcp.tr_iw : jcp.iw;
        sub(reg_input, typesize * iw * inp_mult);
        sub(reg_kernel, typesize * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step_fma(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{

    int kw  = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            Zmm zmm(i_kw * ic_block_step + i_ic);
            if (jcp.is_lastconv) zmm = zmm | KMASK_FLCONV;
            vmovups(zmm, EVEX_compress_addr(reg_kernel, typesize
                    * (i_kw * ic_block + i_ic) * jcp.oc_block + kernel_offset));
        }

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        if (i_ur == 0) {
            Zmm zmm(kw * ic_block_step + (i_ur + 0) % 4);
            if (jcp.is_lastconv) zmm = zmm | KMASK_FLCONV;
            vmovups(zmm, EVEX_compress_addr(reg_output, typesize
                                    * (i_ur + 0) * oc_block + output_offset));
            if (ur_w > 1) {
                Zmm zmm(kw * ic_block_step + (i_ur + 1) % 4);
                if (jcp.is_lastconv) zmm = zmm | KMASK_FLCONV;
                vmovups(zmm, EVEX_compress_addr(reg_output, typesize
                                    * (i_ur + 1) * oc_block + output_offset));
            }
            if (ur_w > 2) {
                Zmm zmm(kw * ic_block_step + (i_ur + 2) % 4);
                if (jcp.is_lastconv) zmm = zmm | KMASK_FLCONV;
                vmovups(zmm, EVEX_compress_addr(reg_output, typesize
                                    * (i_ur + 2) * oc_block + output_offset));
            }
            if (ur_w > 3) {
                Zmm zmm(kw * ic_block_step + (i_ur + 3) % 4);
                if (jcp.is_lastconv) zmm = zmm | KMASK_FLCONV;
                vmovups(zmm, EVEX_compress_addr(reg_output, typesize
                                    * (i_ur + 3) * oc_block + output_offset));
            }
        } else if (i_ur + 3 < ur_w) {
            Zmm zmm(kw * ic_block_step + (i_ur + 3) % 4);
            if (jcp.is_lastconv) zmm = zmm | KMASK_FLCONV;
            vmovups(zmm, EVEX_compress_addr(reg_output, typesize
                                    * (i_ur + 3) * oc_block + output_offset));
        }

        for (int  i_kw = 0; i_kw < kw; i_kw++) {
            int i_iw = i_ur * jcp.stride_w + i_kw;
            if (i_iw - pad_l < 0 || i_iw > (ur_w - 1) * jcp.stride_w + kw - 1
                - pad_r) continue;
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                const int i_offset = input_offset
                    + typesize * (jcp.ver == ver_4fma
                            ? (i_iw - pad_l + i_ic * jcp.tr_iw)
                            : (jcp.is_1stconv
                                ? (i_iw - pad_l) + i_ic * (jcp.ih * jcp.iw)
                                : (i_iw - pad_l) * ic_block + i_ic));
                Zmm zmm(i_kw * ic_block_step + i_ic);
                if (jcp.is_lastconv) zmm = zmm | KMASK_FLCONV;
                vfmadd231ps(zmm, Zmm(kw * ic_block_step + i_ur % 4),
                    EVEX_compress_addr(reg_input, i_offset, true));
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            Zmm zmm(i_kw * ic_block_step + i_ic);
            if (jcp.is_lastconv) zmm = zmm | KMASK_FLCONV;
            vmovups(EVEX_compress_addr(reg_kernel, typesize
                * (i_kw * ic_block + i_ic) * jcp.oc_block + kernel_offset), zmm);
        }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step_4fma(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{
    // TODO: add prefetches to fma version as well

    assert(jcp.ver == ver_4fma);

    int kw  = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    auto zmm_ker = [=](int i_kw, int i_ic) {
        return Zmm(i_kw * ic_block_step + i_ic);
    };

    auto ker_addr = [=](int i_kw, int i_ic) {
        size_t local_offset
            = typesize * (i_kw * ic_block + i_ic) * jcp.oc_block;
        return EVEX_compress_addr(reg_kernel, local_offset + kernel_offset);
    };

    auto inp_addr = [=](int i_iw, int i_ic, ptrdiff_t extra_offset = 0) {
        int stride = jcp.tr_iw * (jcp.is_1stconv ? jcp.ih : 1);
        int local_offset = typesize * (i_iw + i_ic * stride);
        return EVEX_compress_addr(reg_input,
                local_offset + input_offset + extra_offset);
    };

    auto zmm_out = [=](int i_iw) {
        // TODO: move reg calc to global member funcs
        const int out_zmm_base_idx = 28;
        return Zmm(out_zmm_base_idx + i_iw % 4);
    };

    auto out_addr = [=](int i_ur) {
        return EVEX_compress_addr(reg_output,
                typesize * i_ur * oc_block + output_offset);
    };

    auto pf_callback = [=](int i_ur, int i_kw, int i_ic) {
        assert(i_ur % 4 == 0);
        if (i_ur == 0)
            prefetcht1(ker_addr(i_kw, i_ic));
        if (i_ur + 4 >= ur_w)
            prefetcht0(ker_addr(i_kw, i_ic));

        const ptrdiff_t next_input_block_offset
            = typesize * ic_block_step * jcp.tr_iw;
        if (i_ur % 16 == 4 && i_kw == 0) {
            if (i_ur + 16 < ur_w)
                prefetcht0(inp_addr(i_ur + 16, i_ic));
            else
                prefetcht0(inp_addr(0, i_ic, next_input_block_offset));
        }
        if (i_ur % 16 == 4 && i_kw == 1) {
            if (input_wraparound)
                prefetcht1(inp_addr(i_ur, i_ic, -input_offset));
            else
                prefetcht1(inp_addr(i_ur, i_ic, next_input_block_offset));
        }
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto zmm = zmm_ker(i_kw, i_ic);
            vpxord(zmm, zmm, zmm);
        }

    for (int i_ur = 0; i_ur < ur_w; i_ur += 4) {

        for (int i = 0; i < 4; i++) {
            auto zmm = zmm_out(i_ur + i);
            if (i_ur + i < ur_w)
                vmovups(zmm, out_addr(i_ur + i));
            else
                vpxord(zmm, zmm, zmm);
            prefetcht0(out_addr(i_ur + i + 4));
        }

        for (int  i_kw = 0; i_kw < kw; i_kw++)
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                int i_iw = i_ur + i_kw;
                v4fmaddps(zmm_ker(i_kw, i_ic),
                        zmm_out(i_ur), inp_addr(i_iw, i_ic));
                pf_callback(i_ur, i_kw, i_ic);
            }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto addr = ker_addr(i_kw, i_ic);
            auto zmm = zmm_ker(i_kw, i_ic);
            vaddps(zmm, zmm, addr);
            vmovups(addr, zmm);
        }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{
    if (jcp.ver == ver_4fma)
        compute_ic_block_step_4fma(ur_w, pad_l, pad_r,
                ic_block_step, input_offset, kernel_offset, output_offset,
                input_wraparound);
    else if (jcp.ver == ver_fma)
        compute_ic_block_step_fma(ur_w, pad_l, pad_r,
                ic_block_step, input_offset, kernel_offset, output_offset,
                input_wraparound);
    else
        assert(!"unknown convolution version");
}


void
jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow_icblock(
    int ic_block_step, int max_ur_w)
{
    UNUSED(max_ur_w);

    Label kh_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int iw = jcp.ver == ver_4fma ? jcp.tr_iw : jcp.iw;

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - 1
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    mov(kj, reg_kh);
    L(kh_label);
    {
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
            const int input_offset = typesize
                * (jcp.ver == ver_4fma ? i_b_ic * iw : i_b_ic);
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                input_offset, typesize * i_b_ic * jcp.oc_block, 0,
                i_b_ic + ic_block_step >= jcp.ic_block);
        }
        add(reg_input, typesize * iw * inp_mul);
        add(reg_kernel, typesize * (jcp.kw) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow(
    int ic_block_step, int max_ur_w)
{
    Label kh_label, ic_block_label;

    UNUSED(max_ur_w);

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int r_pad = nstl::max(0,
        (jcp.ow - 1) * jcp.stride_w + jcp.kw - 1
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    mov(kj, reg_kh);
    L(kh_label);
    {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            compute_ic_block_step(jcp.ow, l_pad, r_pad, ic_block_step,
                0, 0, 0);
            int inp_icblk_stride = jcp.is_1stconv
                ? jcp.ih * jcp.iw
                : (jcp.ver == ver_4fma ? jcp.tr_iw : 1);
            add(reg_input, typesize * ic_block_step * inp_icblk_stride);
            add(reg_kernel,  typesize * ic_block_step * oc_block);
            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }

        if (jcp.is_1stconv) {
            sub(reg_input, typesize * jcp.ih * jcp.iw * ic_block);
            add(reg_input, typesize * jcp.iw);
        } else if (jcp.ver != ver_4fma) {
            add(reg_input, typesize * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel,  typesize * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_step_common(
    int ic_block_step, int max_ur_w)
{
    Label kh_label, ic_block_label, ow_block_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - 1
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.ver == ver_4fma ? 0 : jcp.l_pad;

    int ur_w     = nstl::min(jcp.ow, max_ur_w);
    int ur_w_trips = jcp.ow / ur_w;
    int ur_w_tail  = jcp.ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0)
        || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }

    int inp_mult = (jcp.is_1stconv || jcp.ver == ver_4fma) ? 1 : ic_block;
    int input_comeback = (ur_w_trips * ur_w * jcp.stride_w - l_pad) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * oc_block;

    mov(kj, reg_kh);
    L(kh_label); {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            if (l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_input, typesize * (ur_w * jcp.stride_w - l_pad)
                    * inp_mult);
                add(reg_output, typesize * ur_w * oc_block);
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                L(ow_block_label); {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_input, typesize * ur_w * jcp.stride_w * inp_mult);
                    add(reg_output, typesize * ur_w * oc_block);

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_trips);
                    jl(ow_block_label, T_NEAR);
                }
            }

            if (ur_w_tail > 0) compute_ic_block_step(ur_w_tail, 0, r_pad,
                ic_block_step, 0, 0, 0);

            sub(reg_input, typesize * input_comeback);
            sub(reg_output, typesize * output_comeback);
            int inp_icblk_stride = jcp.is_1stconv
                ? jcp.ih * jcp.iw
                : (jcp.ver == ver_4fma ? jcp.tr_iw : 1);
            add(reg_input, typesize * ic_block_step * inp_icblk_stride);
            add(reg_kernel, typesize * ic_block_step * oc_block);

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
        if (jcp.is_1stconv) {
            sub(reg_input, typesize * jcp.ih * jcp.iw * ic_block);
            add(reg_input, typesize * jcp.iw);
        } else if (jcp.ver != ver_4fma) {
            add(reg_input, typesize * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, typesize * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_step_disp()
{
    int ic_block_step = jcp.kw <= 3 ? 8 : (jcp.kw <= 7 ? 4 : 2);
    if (jcp.is_1stconv) {
        bool large_code = jcp.kw >= 7 && (jcp.l_pad > 0 || jcp.t_pad > 0);
        ic_block_step
            = (jcp.kw * jcp.ic_block <= 28 && !large_code) ? jcp.ic_block : 1;
    }

    bool too_large_to_unroll
        = (jcp.kw > 1 || jcp.kh > 1) && (jcp.stride_w > 1 || jcp.stride_h > 1);

    if (jcp.kw <= 3 && jcp.ow <= 16 && !too_large_to_unroll)
        compute_oh_step_unroll_ow_icblock(ic_block_step, max_ur_w);
    else if (jcp.ow <= max_ur_w)
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    oh_step_comeback_pointers();
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::maybe_zero_kernel()
{
    Label skip_zeroing, zeroing_loop;

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jz(skip_zeroing, T_NEAR);

    Zmm zero = Zmm(0);
    if (jcp.is_lastconv) zero = zero | KMASK_FLCONV;

    vpxord(zero, zero, zero);
    xor_(reg_tmp, reg_tmp);
    L(zeroing_loop); {
        assert(jcp.is_lastconv ||
               jcp.oc_block * typesize == cpu_isa_traits<avx512_common>::vlen);
        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
            vmovups(ptr[reg_kernel + reg_tmp + ic1 * jcp.oc_block * typesize],
                    zero);
        add(reg_tmp, jcp.ic_block * jcp.oc_block * typesize);
        cmp(reg_tmp, jcp.ic_block * jcp.oc_block * jcp.kw * jcp.kh * typesize);
        jnz(zeroing_loop);
    }

    L(skip_zeroing);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_loop_common()
{
    int b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - 1
        - (jcp.ih + jcp.t_pad - 1));
    int t_pad = jcp.t_pad;
    int stride_h = jcp.stride_h;
    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int iw = jcp.ver == ver_4fma ? jcp.tr_iw : jcp.iw;
    Label oh_label, oh_label_end, oh_tpad_label, oh_bpad_label,
        oh_bpad_label_end;

    maybe_zero_kernel();

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    if (t_pad > 0) {
        assert(jcp.kh <= t_pad + jcp.ih); /* [bwd_w:r1] */
        mov(reg_kh, jcp.kh <= t_pad + jcp.ih ? jcp.kh - t_pad : jcp.ih);
        add(reg_kernel, typesize * t_pad * jcp.kw * jcp.ic_block
            * jcp.oc_block);

        L(oh_tpad_label); {
            compute_oh_step_disp();
            add(reg_output, typesize * jcp.ow * jcp.oc_block);
            sub(reg_kernel, typesize * stride_h * jcp.kw * jcp.ic_block
                * jcp.oc_block);

            inc(reg_oj);
            add(reg_ih_count, stride_h);
            add(reg_kh, stride_h);

            /* the overlap between input and kernel may not reach kernel size.
             * so far we do not support that (until we put constant here) */
            const int final_inp_ker_overlap = jcp.kh; /* [bwd_w:r2] */
            cmp(reg_kh, final_inp_ker_overlap);
            jl(oh_tpad_label, T_NEAR);
        }
        if (t_pad % stride_h != 0) {
            int inp_corr = stride_h -  t_pad % stride_h;
            add(reg_kernel, typesize * inp_corr * jcp.kw * jcp.ic_block
                * jcp.oc_block);
            add(reg_input, typesize * inp_corr * iw * inp_mult);
        }

    }

    cmp(reg_ih_count, jcp.ihp - b_pad - jcp.kh + 1);
    jge(oh_label_end, T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(oh_label, T_NEAR);

    mov(reg_kh, jcp.kh);
    L(oh_label); {
        compute_oh_step_disp();
        add(reg_input, typesize * stride_h * iw * inp_mult);
        add(reg_output, typesize * jcp.ow * jcp.oc_block);

        inc(reg_oj);
        add(reg_ih_count, stride_h);

        cmp(reg_ih_count, jcp.ihp - b_pad - jcp.kh + 1);
        jge(oh_label_end, T_NEAR);

        cmp(reg_oj, jcp.oh);
        jl(oh_label, T_NEAR);
    }
    L(oh_label_end);

    if (b_pad > 0) {
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_label_end, T_NEAR);

        mov(reg_kh,  jcp.ihp - b_pad);
        sub(reg_kh, reg_ih_count);
        L(oh_bpad_label);
        {
            compute_oh_step_disp();
            add(reg_input, typesize * stride_h * iw * inp_mult);
            add(reg_output, typesize * jcp.ow * jcp.oc_block);

            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(oh_bpad_label_end, T_NEAR);

            inc(reg_oj);
            cmp(reg_oj, jcp.oh);
            jl(oh_bpad_label, T_NEAR);
        }
        L(oh_bpad_label_end);
    }
}

bool jit_avx512_common_conv_bwd_weights_kernel_f32::compute_full_spat_loop()
{
    // FIXME: use register mapping from the class declaration
    if (jcp.ver != ver_4fma || jcp.stride_h != 1 || jcp.stride_w != 1)
        return false;

    if (jcp.l_pad != jcp.kw / 2 || jcp.t_pad != jcp.kh / 2)
        return false;

    // General code layout:
    //
    // Blocking over OH -- top level
    // (Reduces L2 pressure; not very useful right now)
    //  Loop over all KHxKW kernel -- emit_kh_kw_loop()
    //    Loop over OH block -- emit_h_loop()
    //      Loop over OW blocks -- emit_fma_block()
    //      (Supports both fully unrolled and partially unrolled versions to
    //      reduce code size)
    //          Loop over OW block -- emit_fma_step()

    int max_working_set_size = 128 * 1024;

    int inp_row_size = jcp.ic_block * jcp.tr_iw * typesize;
    int out_row_size = jcp.oc_block * jcp.ow * typesize;
    int row_size = inp_row_size + out_row_size;

    int h_block_size = jcp.oh;
    int working_set_size = row_size * h_block_size;

    if (working_set_size > max_working_set_size) {
        int opt_working_set_size = 48 * 1024;
        assert(opt_working_set_size < max_working_set_size);

        while (working_set_size > opt_working_set_size) {
            for (int i = 2; i <= h_block_size; i++)
                if (i == h_block_size)
                    h_block_size = h_block_size / 2;
                else if (h_block_size % i == 0) {
                    h_block_size = h_block_size / i;
                    break;
                }
            working_set_size = row_size * h_block_size;

            if (h_block_size == 1 && working_set_size > opt_working_set_size)
                return false;
        }
    }

    if (h_block_size < nstl::max(1, jcp.t_pad))
        return false;

    // check that we can use simple arithmetic for prefetch address
    // calculations
    // TODO: we need some traits for this check (Roma)
    int cache_line_size = 64;
    assert(jcp.ic_block * typesize == 64);
    assert(jcp.oc_block * typesize == 64);

    int num_inp_l2_pfs = jcp.tr_iw * h_block_size;
    int avg_h_loop_len = h_block_size;
    int num_inp_l2_pfs_per_fma_block
        = div_up(num_inp_l2_pfs, avg_h_loop_len * jcp.kw * jcp.kh);
    int num_out_l2_pfs = jcp.ow * h_block_size;
    int num_out_l2_pfs_per_fma_block
        = div_up(num_out_l2_pfs, avg_h_loop_len * jcp.kw * jcp.kh);

    Opmask reg_h_block = k1; // 32-bit only on Intel(R) Xeon Phi(TM) processors
    Reg64 reg_kh = rax;
    Reg64 reg_kw = rbx;
    Reg64 reg_tmp = abi_not_param1;
    Reg32 reg_tmp_w = reg_tmp.cvt32();
    Reg64 reg_ohs = rdx;
    Reg64 reg_ihs = rsi;
    Reg64 reg_h = r8;
    Reg64 reg_i = r9;
    Reg64 reg_j = r10;

    Reg64 reg_inp = r13;
    Reg64 reg_out = r14;
    Reg64 reg_ker = r15;

    Reg64 reg_inp_pf_l1 = rbp;

    Reg64 reg_inp_pf_l2 = r11;
    Reg64 reg_out_pf_l2 = r12;

    Xmm reg_inp_pf_save = xmm17;
    Xmm reg_out_pf_save = xmm18;

    Reg64 reg_inp_save = abi_param1;
    Reg64 reg_out_save = reg_tmp;

    auto zmm_out = [&](int oi) { return Zmm(24 + oi % 8); };
    auto zmm_ker = [&](int ic1) { return Zmm(ic1); };
    auto inp_addr = [&](int oi, int ic1) {
        return ptr[reg_inp + (ic1 * jcp.tr_iw + oi) * typesize];
    };
    auto out_addr = [&](int oi, int oj = 0) {
        return ptr[reg_out + ((oi + oj * jcp.ow) * jcp.oc_block) * typesize];
    };
    auto ker_addr = [&](int ic1) {
        return ptr[reg_ker + ic1 * jcp.oc_block * typesize];
    };

    auto emit_fma_block = [&](int h_block_size,
            bool is_last_block, bool is_last_kh_kw_iter, bool is_last_row)
    {
        // TODO: add an fma version (Roma)

        int ow4u = rnd_up(jcp.ow, 4);
        int def_step_size = 16;

        bool has_w_tail = (jcp.ow % def_step_size != 0 || jcp.ow % 4 != 0);
        bool full_w_unroll = jcp.ow / def_step_size < 2 + has_w_tail;

        auto emit_fma_step = [&](int step_size,
                int num_inp_l1_pfs_per_fma_step,
                int num_inp_l2_pfs_per_fma_step,
                int num_out_l2_pfs_per_fma_step, bool is_w_tail)
        {
            bool block_wraparound = is_w_tail && is_last_row;
            bool block_end = block_wraparound && is_last_kh_kw_iter;
            bool the_end = block_end && is_last_block;

            assert(step_size % 4 == 0);
            int tail_size = ow4u % step_size;
            int this_step_size
                = (is_w_tail && tail_size) ? tail_size : step_size;
            int ow_last_chunk4 = jcp.ow % 4;
            int ow_zero_tail4 = ow_last_chunk4 ? 4 - ow_last_chunk4 : 0;

            auto emit_out_pf = [&](int oi) {
#if 1
                if (oi + def_step_size < step_size || !block_wraparound)
                    prefetcht0(ptr[reg_out
                            + ((def_step_size + oi)
                                * jcp.oc_block * typesize)]);
                else {
                    assert(block_wraparound);
                    assert(oi + def_step_size >= step_size);
                    prefetcht0(ptr[reg_out_save
                            + ((oi + def_step_size - step_size)
                                * jcp.oc_block * typesize)]);
                }
#else
                // XXX: This is an alternative prefetching strategy that
                // always prefetches the next row. Keeping it here for
                // future experiments (Roma)
                if (!block_wraparound)
                    prefetcht0(ptr[reg_out
                            + (jcp.ow + oi) * jcp.oc_block * typesize]);
                else
                    prefetcht0(ptr[reg_out + reg_ohs
                            - ((h_block_size - 1) * jcp.ow
                                - oi) * jcp.oc_block * typesize]);
#endif
                if (oi < num_out_l2_pfs_per_fma_step)
                    prefetcht1(ptr[reg_out_pf_l2
                            + oi * jcp.oc_block * typesize]);
            };

            auto emit_inp_pf = [&](int oi4, int ic1) {
                int pf_slot_idx = ic1 + oi4 / 4 * jcp.ic_block;
                int num_pf_slots = jcp.ic_block * step_size / 4;

                int num_pfs = num_inp_l1_pfs_per_fma_step
                    + num_inp_l2_pfs_per_fma_step;
                int pf_freq = nstl::max(1, num_pf_slots / num_pfs);

                if (pf_slot_idx % pf_freq)
                    return;

                int pf_idx = pf_slot_idx / pf_freq;

                if (pf_idx < num_inp_l2_pfs_per_fma_step)
                    prefetcht1(ptr[reg_inp_pf_l2
                            + pf_idx * jcp.ic_block * typesize]);
                else {
                    pf_idx -= num_inp_l2_pfs_per_fma_step;
                    // Prefetch the 'tail' of the cache line because most of
                    // the accesses are not aligned
                    prefetcht0(ptr[reg_inp_pf_l1
                            + pf_idx * jcp.ic_block * typesize
                            + cache_line_size - typesize]);
                }
            };

            for (int oi4 = 0; oi4 < this_step_size; oi4 += 4) {
                for (int oi1 = 0; oi1 < 4; oi1++) {
                    int oi = oi4 + oi1;
                    if (!is_w_tail || oi < this_step_size - ow_zero_tail4) {
                        vmovups(zmm_out(oi), out_addr(oi));
                        emit_out_pf(oi);
                    } else {
                        auto zmm = zmm_out(oi);
                        vpxord(zmm, zmm, zmm);
                    }
                }

                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    v4fmaddps(zmm_ker(ic1), zmm_out(oi4), inp_addr(oi4, ic1));
                    emit_inp_pf(oi4, ic1);
                }
            }
        };

        // Input is transposed and padded but we only access about jcp.iw
        // elements so use that to compute the # of cache lines in each 'row'
        int num_inp_l1_pfs
            = div_up(jcp.iw * typesize, cache_line_size) * jcp.ic_block;

        if (full_w_unroll) {
            emit_fma_step(ow4u, num_inp_l1_pfs,
                    num_inp_l2_pfs_per_fma_block,
                    num_out_l2_pfs_per_fma_block, true);
            add(reg_inp_pf_l2, num_inp_l2_pfs_per_fma_block * cache_line_size);
            add(reg_out_pf_l2, num_out_l2_pfs_per_fma_block * cache_line_size);
        } else {
            Label w_loop;
            int num_w_iters = jcp.ow / def_step_size;
            int num_w_iters_full = num_w_iters + has_w_tail;
            int num_inp_l1_pfs_per_fma_step
                = div_up(num_inp_l1_pfs, num_w_iters_full);
            int num_inp_l2_pfs_per_fma_step
                = div_up(num_inp_l2_pfs_per_fma_block, num_w_iters_full);
            int num_out_l2_pfs_per_fma_step
                = div_up(num_out_l2_pfs_per_fma_block, num_w_iters_full);
            mov(reg_i, num_w_iters);
            L(w_loop); {
                emit_fma_step(def_step_size, num_inp_l1_pfs_per_fma_step,
                        num_inp_l2_pfs_per_fma_step,
                        num_out_l2_pfs_per_fma_step, false);
                add(reg_inp, def_step_size * typesize);
                add(reg_out, def_step_size * jcp.oc_block * typesize);
                add(reg_inp_pf_l1,
                        num_inp_l1_pfs_per_fma_step * cache_line_size);
                add(reg_inp_pf_l2,
                        num_inp_l2_pfs_per_fma_step * cache_line_size);
                add(reg_out_pf_l2,
                        num_out_l2_pfs_per_fma_step * cache_line_size);
                sub(reg_i, 1);
                jnz(w_loop);
            }
            if (has_w_tail) {
                emit_fma_step(def_step_size, num_inp_l1_pfs_per_fma_step,
                        num_inp_l2_pfs_per_fma_step,
                        num_out_l2_pfs_per_fma_step, true);
                add(reg_inp_pf_l2,
                        num_inp_l2_pfs_per_fma_step * cache_line_size);
                add(reg_out_pf_l2,
                        num_out_l2_pfs_per_fma_step * cache_line_size);
            }
            // reset reg_inp and reg_out because emit_h_loop expects
            // unmodified pointers
            int w_offset = num_w_iters * def_step_size;
            sub(reg_inp, w_offset * typesize);
            sub(reg_out, w_offset * jcp.oc_block * typesize);
        }
    };

    auto emit_h_loop = [&](int h_block_size,
            bool is_last_block, bool is_last_kh_kw_iter)
    {
        Label h_loop, skip_h_loop;
        mov(reg_j, 1);
        cmp(reg_j, reg_h);
        je(skip_h_loop, T_NEAR);
        L(h_loop); {

            lea(reg_inp_pf_l1,
                    ptr[reg_inp + jcp.tr_iw * jcp.ic_block * typesize]);
            emit_fma_block(h_block_size,
                    is_last_block, is_last_kh_kw_iter, false);

            add(reg_inp, jcp.tr_iw * jcp.ic_block * typesize);
            add(reg_out, jcp.ow * jcp.oc_block * typesize);
            add(reg_j, 1);
            cmp(reg_j, reg_h);
            jb(h_loop);
        }

        L(skip_h_loop);

        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
            prefetcht0(ker_addr(ic1));

        lea(reg_inp_pf_l1, ptr[reg_inp_save + reg_kw * typesize]);
        emit_fma_block(h_block_size, is_last_block, is_last_kh_kw_iter, true);
    };

    auto emit_kh_kw_loop = [&](bool is_first_block, bool is_last_block,
            int h_block_size)
    {
        xor_(reg_kh, reg_kh);
        Label kh_loop, kh_loop_end;

        int last_oh_block_size
            = jcp.oh - rnd_up(jcp.oh - h_block_size, h_block_size);
        int oh_block_size = (is_last_block) ? last_oh_block_size : h_block_size;
        // NB: this is correct because we only support t_pad = kh / 2 and thus
        // ih == oh
        int ih_block_size
            = oh_block_size + (!is_first_block + !is_last_block) * jcp.t_pad;

        L(kh_loop); {
            if (is_first_block) {
                xor_(reg_tmp, reg_tmp);
                mov(reg_ohs, jcp.t_pad);
                sub(reg_ohs, reg_kh);
                cmovb(reg_ohs, reg_tmp);

                mov(reg_ihs, reg_ohs);
                sub(reg_ihs, jcp.t_pad);
                add(reg_ihs, reg_kh);
            } else {
                xor_(reg_ohs, reg_ohs);
                mov(reg_ihs, reg_kh);
            }

            mov(reg_tmp, oh_block_size);
            sub(reg_tmp, reg_ohs);
            mov(reg_h, ih_block_size);
            sub(reg_h, reg_ihs);
            cmp(reg_tmp, reg_h);
            cmovb(reg_h, reg_tmp);

            Label kh_loop_work;
            cmp(reg_h, 0);
            jg(kh_loop_work, T_NEAR);

            // empty h loop for this jcp.kh:
            // - set the output to 0 if necessary
            // - move ker pt
            // - jump to the end
            sub(reg_h, 1);
            Label skip_ker_zeroing;

            // The reg_ker ptr has highest bit set if the output needs to be
            // zeroed. Those who have byte-aligned their data will suffer the
            // consiquences :(
            // TODO: move the flag to a mask register? (Roma)
            test(reg_tmp, 1);
            jnz(skip_ker_zeroing, T_NEAR);

            Label zeroing_loop;
            vpxord(zmm0, zmm0, zmm0);
            and_(reg_ker, ~1); // temporarily clear the zeroing flag
            mov(reg_tmp, jcp.kw);
            L(zeroing_loop); {
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
                    vmovups(ker_addr(ic1), zmm0);
                add(reg_ker, jcp.oc_block * jcp.ic_block * typesize);
                sub(reg_tmp, 1);
                jnz(zeroing_loop, T_NEAR);
            }
            // restore the zeroing flag (it will be cleared after the end of
            // emit_kh_kw_loop, but we may need it until then)
            or_(reg_ker, 1);
            jmp(kh_loop_end, T_NEAR);

            L(skip_ker_zeroing);
            add(reg_ker, jcp.oc_block * jcp.ic_block * jcp.kw * typesize);
            jmp(kh_loop_end, T_NEAR);

            L(kh_loop_work);

            mul_by_const(reg_ihs, reg_tmp, jcp.tr_iw * jcp.ic_block * typesize);
            mul_by_const(reg_ohs, reg_tmp, jcp.ow * jcp.oc_block * typesize);

            add(reg_inp, reg_ihs);
            add(reg_out, reg_ohs);

            Label kw_loop;
            xor_(reg_kw, reg_kw);
            L(kw_loop); {
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    auto zmm = zmm_ker(ic1);
                    vpxord(zmm, zmm, zmm);
                    prefetcht1(ker_addr(ic1));
                }

                mov(reg_out_save, reg_out);
                mov(reg_inp_save, reg_inp);
                lea(reg_inp, ptr[reg_inp + reg_kw * typesize]);

#if 0
                // XXX: Generate code with special prefetches when switching
                // blocks or at the end of the last block. Disabled to reduce
                // code size and because there's no performance benefit (Roma)
                Label regular_h_loop, end_h_loop;
                cmp(reg_kw, jcp.kw - 1);
                jne(regular_h_loop, T_NEAR);
                cmp(reg_kh, jcp.kh - 1);
                jne(regular_h_loop, T_NEAR);

                emit_h_loop(oh_block_size, is_last_block, true);
                jmp(end_h_loop, T_NEAR);

                L(regular_h_loop);
                emit_h_loop(oh_block_size, is_last_block, false);

                L(end_h_loop);
#else
                emit_h_loop(oh_block_size, is_last_block, false);
#endif

                mov(reg_out, reg_out_save);
                mov(reg_inp, reg_inp_save);

                Label do_store;
                // The reg_ker ptr has highest bit set if the output needs to
                // be zeroed. Those who have byte-aligned their data will
                // suffer the consiquences :(
                mov(reg_tmp, reg_ker);
                and_(reg_ker, ~1);
                test(reg_tmp, 1);
                jnz(do_store, T_NEAR);

                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
                    vaddps(zmm_ker(ic1), ker_addr(ic1));

                L(do_store);
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
                    vmovups(ker_addr(ic1), zmm_ker(ic1));

                mov(reg_ker, reg_tmp);
                add(reg_ker, jcp.ic_block * jcp.oc_block * typesize);
                add(reg_kw, 1);
                cmp(reg_kw, jcp.kw);
                jl(kw_loop);
            }

            sub(reg_inp, reg_ihs);
            sub(reg_out, reg_ohs);


            L(kh_loop_end);
            add(reg_kh, 1);
            cmp(reg_kh, jcp.kh);
            jl(kh_loop);
        }
    };

    mov(reg_inp, ptr[param + GET_OFF(src)]);
    mov(reg_out, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);
    mov(reg_inp_pf_l2, ptr[param + GET_OFF(src_prf)]);
    mov(reg_out_pf_l2, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    or_(reg_ker, reg_tmp);

    bool single_kh_kw_loop = (h_block_size == jcp.oh);

    size_t inp_row_step = jcp.tr_iw * jcp.ic_block * typesize;
    size_t first_inp_block_step = inp_row_step * (h_block_size - jcp.t_pad);
    size_t inp_block_step = inp_row_step * h_block_size;
    size_t out_block_step = jcp.ow * jcp.oc_block * typesize * h_block_size;

    if (!single_kh_kw_loop) {
        // Save the original prefetch pointers from the OpenMP driver
        vmovq(reg_inp_pf_save, reg_inp_pf_l2);
        vmovq(reg_out_pf_save, reg_out_pf_l2);
        mov(reg_inp_pf_l2, reg_inp);
        add(reg_inp_pf_l2, first_inp_block_step);
        mov(reg_out_pf_l2, reg_out);
        add(reg_out_pf_l2, out_block_step);
    }
    emit_kh_kw_loop(true, single_kh_kw_loop, h_block_size);

    if (!single_kh_kw_loop) {
        size_t ker_reset_offset
            = jcp.oc_block * jcp.ic_block * typesize * jcp.kw * jcp.kh;
        sub(reg_ker, ker_reset_offset);
        and_(reg_ker, ~1); // Clear the zeroing flag for subsequent updates

        add(reg_inp, first_inp_block_step);
        add(reg_out, out_block_step);
        mov(reg_inp_pf_l2, reg_inp);
        add(reg_inp_pf_l2, inp_block_step);
        mov(reg_out_pf_l2, reg_out);
        add(reg_out_pf_l2, out_block_step);

        int num_innermost_iters = div_up(jcp.oh, h_block_size) - 2;
        if (num_innermost_iters > 0) {
            bool need_innermost_loop = num_innermost_iters > 1;
            Label h_block_loop;

            mov(reg_tmp_w, num_innermost_iters);
            kmovw(reg_h_block, reg_tmp_w);
            L(h_block_loop); {
                emit_kh_kw_loop(false, false, h_block_size);
                sub(reg_ker, ker_reset_offset);
                add(reg_inp, inp_row_step * h_block_size);
                add(reg_out, out_block_step);
                mov(reg_inp_pf_l2, reg_inp);
                add(reg_inp_pf_l2, inp_block_step);
                mov(reg_out_pf_l2, reg_out);
                add(reg_out_pf_l2, out_block_step);
                kmovw(reg_tmp_w, reg_h_block);
                sub(reg_tmp_w, 1);
                kmovw(reg_h_block, reg_tmp_w);
                jnz(h_block_loop);
            }
        }

        // Restore the original prefetch pointers that came from the OpenMP
        // driver
        vmovq(reg_inp_pf_l2, reg_inp_pf_save);
        vmovq(reg_out_pf_l2, reg_out_pf_save);
        emit_kh_kw_loop(false, true, h_block_size);
    }

    return true;
}

bool jit_avx512_common_conv_bwd_weights_kernel_f32::flat_4ops_compute() {
    const auto &j = jcp;
    const bool ok = j.ver == ver_4fma && j.is_1stconv;
    if (!ok) return false;

    Reg64 reg_ptr_tr_src = r8;
    Reg64 reg_ptr_dst = r9;
    Reg64 reg_ptr_wei = r10;
    Reg64 reg_ptr_bia = r11;

    Reg64 reg_kh_step = rax;
    Reg64 reg_oh = abi_not_param1;
    Reg64 reg_kh = rdx;

    Reg32 reg_flag_save = ebx;
    Reg32 reg_flag = esi;

    Zmm vbia(31);

    auto zmm_wei = [&](int kh, int kw) {
        return Zmm(8 + kh * j.kw + kw);
    };
    auto zmm_dst = [&](int ow) {
        return Zmm(ow % 8);
    };

    auto addr_tr_src = [&](int kh, int iw) {
        return ptr[reg_ptr_tr_src
            + (kh * j.stride_w * j.tr_ld + iw) * typesize];
    };
    auto addr_dst = [&](int ow) {
        return ptr[reg_ptr_dst + ow * jcp.oc_block * typesize];
    };
    auto addr_wei = [&](int kh, int kw) {
        return ptr[reg_ptr_wei + (kh * j.kw + kw) * j.oc_block * typesize];
    };

    auto emit_fma_block = [&](int kh_step) {
        for (int kh = 0; kh < kh_step; ++kh) {
            for (int kw = 0; kw < j.kw; ++kw) {
                auto vwei = zmm_wei(kh, kw);
                vpxord(vwei, vwei, vwei);
            }
        }

        for (int ow = 0; ow < j.ow; ow += 4) {
            for (int _ow = ow; _ow < ow + 4; ++_ow) {
                auto vdst = zmm_dst(_ow);
                if (_ow < j.ow)
                    vmovups(vdst, addr_dst(_ow));
                else
                    vpxord(vdst, vdst, vdst);
            }

            for (int kh = 0; kh < kh_step; ++kh) {
                for (int kw = 0; kw < j.kw; ++kw) {
                    const int iw = ow + (kw % j.stride_w) * j.tr_ld
                        + (kw / j.stride_w);
                    v4fmaddps(zmm_wei(kh, kw), zmm_dst(ow),
                            addr_tr_src(kh, iw));
                    if (1 && kh == 0 && kw < 4) {
                        prefetcht1(ptr[reg_ptr_dst
                                + (j.ow + ow + kw) * jcp.oc_block * typesize]);
                    }
                    if (j.with_bias && kh_step == 1) { /* [bwd_w:b:r1] */
                        const int off = kw + 4 - j.kw;
                        if (off >= 0 && ow + off < j.ow)
                            vaddps(vbia, vbia, zmm_dst(ow + off));
                    }
                }
            }
        }

        Label l_store;
        test(reg_flag, FLAG_MB_FIRST);
        jnz(l_store, T_NEAR);
        for (int kh = 0; kh < kh_step; ++kh) {
            for (int kw = 0; kw < j.kw; ++kw)
                vaddps(zmm_wei(kh, kw), addr_wei(kh, kw));
        }
        L(l_store);
        for (int kh = 0; kh < kh_step; ++kh) {
            for (int kw = 0; kw < j.kw; ++kw)
                vmovups(addr_wei(kh, kw), zmm_wei(kh, kw));
        }
    };

    auto emit_kh_loop = [&]() {
        const int kh_step_rem = j.kh % j.kh_step;
        xor_(reg_kh, reg_kh);
        mov(reg_kh_step, j.kh_step);

        Label l_kh_loop;
        L(l_kh_loop); {
            Label l_done;

            if (kh_step_rem != 0) {
                Label l_keep_kh_step;
                cmp(reg_kh, j.kh - j.kh_step);
                jle(l_keep_kh_step, T_NEAR);

                mov(reg_kh_step, kh_step_rem);
                emit_fma_block(kh_step_rem);
                jmp(l_done, T_NEAR);

                L(l_keep_kh_step);
            }

            emit_fma_block(j.kh_step);

            L(l_done);

            add(reg_ptr_tr_src, j.kh_step * j.stride_w * j.tr_ld * typesize);
            add(reg_ptr_wei, j.kh_step * j.kw * j.oc_block * typesize);
            add(reg_kh, j.kh_step);

            cmp(reg_kh, j.kh);
            jl(l_kh_loop, T_NEAR);
        }

        const int kh_steps = rnd_up(j.kh, j.kh_step);
        sub(reg_ptr_tr_src, kh_steps * j.stride_w * j.tr_ld * typesize);
        sub(reg_ptr_wei, kh_steps * j.kw * j.oc_block * typesize);
    };

    auto emit_oh_loop = [&]() {
        mov(reg_oh, j.oh);

        Label l_oh_loop;
        L(l_oh_loop); {
            Label l_restore_mb_flag, l_jump;

            cmp(reg_oh, j.oh);
            je(l_restore_mb_flag, T_NEAR);

            and_(reg_flag, ~FLAG_MB_FIRST);
            jmp(l_jump, T_NEAR);

            L(l_restore_mb_flag);
            mov(reg_flag, reg_flag_save);

            L(l_jump);

            emit_kh_loop();

            add(reg_ptr_tr_src, j.stride_h * j.stride_w * j.tr_ld * typesize);
            add(reg_ptr_dst, j.ow * j.oc_block * typesize);

            dec(reg_oh);
            jnz(l_oh_loop, T_NEAR);
        }
    };

    auto emit_bia_store = [&]() {
        if (!j.with_bias) return;

        Label l_bia_store, l_bia_skip;
        test(reg_flag, FLAG_IC_FIRST);
        jz(l_bia_skip);

        test(reg_flag, FLAG_MB_FIRST);
        jnz(l_bia_store, T_NEAR);
        vaddps(vbia, ptr[reg_ptr_bia]);
        L(l_bia_store);
        vmovups(ptr[reg_ptr_bia], vbia);
        L(l_bia_skip);
    };

    mov(reg_ptr_tr_src, ptr[param + GET_OFF(src)]);
    mov(reg_ptr_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ptr_wei, ptr[param + GET_OFF(filt)]);
    mov(reg_ptr_bia, ptr[param + GET_OFF(bias)]);
    mov(reg_flag_save, ptr[param + GET_OFF(flags)]);

    vpxord(vbia, vbia, vbia);
    emit_oh_loop();
    emit_bia_store();

    return true;
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_loop()
{
    if (flat_4ops_compute())
        return;
    if (compute_full_spat_loop())
        return;
    compute_oh_loop_common();
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::generate()
{
    preamble();

    if (jcp.is_lastconv && (jcp.ver == ver_fma))
        SET_FLCONV_KMASK(jcp.oc_block);

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    compute_loop();

    postamble();
}

status_t jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf(
    jit_conv_conf_t &jcp, const convolution_desc_t &cd,
    cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &diff_weights_pd,
    cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd)
{
    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper diff_weights_d(&diff_weights_pd);
    const memory_desc_wrapper diff_bias_d(&diff_bias_pd);
    const memory_desc_wrapper diff_dst_d(&diff_dst_pd);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;

    jcp = zero<decltype(jcp)>();
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = diff_weights_d.dims()[with_groups + 2];
    jcp.kw = diff_weights_d.dims()[with_groups + 3];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    if (jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw
        - jcp.l_pad);
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih
        - jcp.t_pad);

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format() == any)
            CHECK(diff_bias_pd.set_format(x));
        if (diff_bias_d.format() != x)
            return status::unimplemented;
    }

    /* conditions on destination memory */
    jcp.is_lastconv = jcp.oc % simd_w;
    if (jcp.is_lastconv) {
        if (one_of(jcp.oc, 3, 4)) { // OC=1 may have performance issue
            if (diff_dst_d.format() == any)
                CHECK(diff_dst_pd.set_format(nhwc));
            if (diff_dst_d.format() != nhwc)
                return status::unimplemented;
            jcp.oc_block = jcp.oc;
        } else
            return status::unimplemented;
    } else {
        jcp.oc_block = simd_w;
        if (jcp.oc % jcp.oc_block)
            return status::unimplemented;
        if (diff_dst_d.format() == any)
            CHECK(diff_dst_pd.set_format(nChw16c));
        if (diff_dst_d.format() != nChw16c)
            return status::unimplemented;
    }
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const bool boundaries_ok = true
        && jcp.t_pad <= jcp.kh / 2
        && jcp.b_pad <= jcp.kh / 2
        && jcp.kh <= jcp.t_pad + jcp.ih /* [bwd_w:r1] */
        && jcp.kh <= jcp.ih; /* [bwd_w:r2] */
    if (!boundaries_ok)
        return status::unimplemented;

    /* yet another common check */
    if (jcp.kw > 14)
        return status::unimplemented;

    /* setting register strategy */
    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) { jcp.ur_w = ur_w; break; }
    }

    /* check for the 1st convolution */
    jcp.is_1stconv = jcp.ic % simd_w;
    if (jcp.is_1stconv) {
        if (src_d.format() == any)
            CHECK(src_pd.set_format(nchw));

        const bool src_ok = true
            && one_of(jcp.ic, 1, 3, 4)
            && implication(jcp.ic == 1, one_of(src_d.format(), nchw, nhwc))
            && implication(jcp.ic != 1, src_d.format() == nchw)
            && jcp.ngroups == 1;
        if (!src_ok)
            return status::unimplemented;

        const int tr_ld = rnd_up(div_up(jcp.iw + jcp.l_pad + jcp.r_pad,
                    jcp.stride_w), 16);
        const int kh_step = nstl::max((28 - jcp.with_bias) / jcp.kw, 1);
        const int kh_step_rem = jcp.kh % kh_step;
        const auto want_4fma_wfmt = with_groups ? gOihw16o : Oihw16o;
        const bool use_4fma = true
            && mayiuse(avx512_mic_4ops)
            && everyone_is(0, jcp.l_pad, jcp.r_pad, jcp.t_pad, jcp.b_pad)
            && jcp.kw <= 28 - jcp.with_bias
            && jcp.stride_w == 4
            && tr_ld / simd_w <= 4 /* [bwd_w:tr_src:r1] */
            && implication(jcp.with_bias, kh_step_rem == 1) /* [bwd_w:b:r1] */
            && implication(diff_weights_d.format() != any,
                    diff_weights_d.format() == want_4fma_wfmt);

        if (use_4fma) {
            jcp.ver = ver_4fma;
            jcp.kh_step = kh_step;
            jcp.tr_ld = tr_ld;
            jcp.ic_block = 1;
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_4fma_wfmt));
        } else {
            jcp.ver = ver_fma;
            jcp.ic_block = jcp.ic;

            const auto want_wfmt = with_groups ? gOhwi16o : Ohwi16o;
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_wfmt));
            if (diff_weights_d.format() != want_wfmt)
                return status::unimplemented;
        }

        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();
    } else {
        if (src_d.format() == any)
            CHECK(src_pd.set_format(nChw16c));
        if (diff_weights_d.format() == any)
            CHECK(diff_weights_pd.set_format(jcp.is_lastconv
                        ? Ihw16io : (with_groups ? gOIhw16i16o : OIhw16i16o)));

        const bool ok = true
            && src_d.format() == nChw16c
            && diff_weights_d.format() == jcp.is_lastconv
                        ? Ihw16io : (with_groups ? gOIhw16i16o : OIhw16i16o);
        if (!ok)
            return status::unimplemented;

        jcp.ic_block = simd_w;
        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();

        if (mayiuse(avx512_mic_4ops) && jcp.stride_w == 1 && !jcp.is_lastconv)
            jcp.ver = ver_4fma;
        else
            jcp.ver = ver_fma;

        if (jcp.ver == ver_4fma) {
            jcp.ur_w = jcp.ow;
            // XXX, BUGBUGBUG, but not a FIXME: this assumes that it's OK to
            // cross the right boundary. The only requirement is not to have
            // NaNs there because another multiplicand is always guaranteed to
            // be zero. This also may require the top-level driver to allocate
            // four extra guarding elements at the very end of the buffer.
            // I'm not proud of this hack, but it improves performance by
            // about 5-10% depending on the dimensions (Roma)
            jcp.tr_iw = rnd_up(jcp.iw + jcp.kw - 1, 4);
            jcp.tr_src_num_guard_elems = 4; // upper bound
        }
    }

    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
