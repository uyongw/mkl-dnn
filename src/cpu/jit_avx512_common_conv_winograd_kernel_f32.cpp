/*******************************************************************************
* Copyright 2017 Intel Corporation
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
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include <math.h>

#include "jit_avx512_common_conv_winograd_kernel_f32.hpp"

#ifndef KERNEL_SIZE_THRESHOLD
#define KERNEL_SIZE_THRESHOLD 16
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

int L1_cache_size = get_cache_size(1, true);
int L2_cache_size = get_cache_size(2, true);

// the test funtion takes jcp, the candidate and the current best.
// it  returns true if the new candidate is better
int get_divisor_satisfying_cond(jit_conv_winograd_conf_t jcp, int number,
        int default_best, bool (*test)(jit_conv_winograd_conf_t, int, int))
{
    int best_divisor = default_best;
    auto test_num
            = [&best_divisor, test](jit_conv_winograd_conf_t jcp, int num) {
                  if (test(jcp, num, best_divisor)) {
                      best_divisor = num;
                  }
              };

    for (int divisor = 1; divisor <= ::sqrt(number); divisor++) {
        if (number % divisor == 0) {
            test_num(jcp, divisor);
            test_num(jcp, number / divisor);
        }
    }

    return best_divisor;
}

// Returns the max divisor of 'number' that satisfying condition specified
// by function 'test'.
int get_max_divisor_satisfying_cond(jit_conv_winograd_conf_t jcp, int number,
        int default_best, bool (*test)(jit_conv_winograd_conf_t, int, int))
{
    int best_divisor = default_best;
    auto test_num
        = [&best_divisor, test](jit_conv_winograd_conf_t jcp, int num) {
            if (test(jcp, num, best_divisor)) {
                best_divisor = num;
                return true;
            }
            return false;
        };

    for (int divisor = 1; divisor <= number; divisor++) {
        if (number % divisor == 0) {
            if (test_num(jcp, number / divisor)) {
                break;
            }
        }
    }

    return best_divisor;
}

// Returns the min divisor of 'number' that satisfying condition specified
// by function 'test'. XXX
int get_min_divisor_satisfying_cond(jit_conv_winograd_conf_t jcp, int number,
        int default_best, bool (*test)(jit_conv_winograd_conf_t, int, int))
{
    int best_divisor = default_best;
    auto test_num
        = [&best_divisor, &number, test](jit_conv_winograd_conf_t jcp, int divisor) {
            if (test(jcp, number, best_divisor)) {
                best_divisor = divisor;
                return true;
            }
            return false;
        };

    for (int divisor = best_divisor + 1; divisor <= number; divisor++) {
        if (number % divisor == 0) {
            if (test_num(jcp, divisor))
                break;
        }
    }

    return best_divisor;
}

/* assumes 512 bits registers */
/* TODO: add support for strides */
/* TODO: handle the prefetch distance automatically */
typedef enum cache_t_ { L1, L2, L3 } cache_t;

template <typename data_t>
struct prefetcher_t {
    prefetcher_t(jit_generator *generator, Xbyak::Reg64 reg_base_addr,
            cache_t cache_type, size_t block_size, /* in number of elements*/
            int nb_instructions_in_block, int fma_ipc)
        : cg_(generator)
        , reg_base_addr_(reg_base_addr)
        , cache_block_size_(block_size)
        , cache_type_(cache_type)
    {
        nb_cache_lines_to_prefetch_ = cache_block_size_ / (64 / sizeof(data_t));
        prefetch_spread_
                = div_up(nb_instructions_in_block, nb_cache_lines_to_prefetch_);
        prefetch_blk_
                = div_up(nb_cache_lines_to_prefetch_, nb_instructions_in_block);

        /* assumption: when fetch in Li, data is already in L(i+1) */
        int cache_latency;
        switch (cache_type_) {
        case L1: cache_latency = 14; break;
        case L2: cache_latency = 250; break;
        case L3: cache_latency = 250; break;
        }

        prefetch_distance_ = div_up(cache_latency, nb_cache_lines_to_prefetch_);
    }

    void prefetch(int instruction_number)
    {
        if (instruction_number % prefetch_spread_ == 0) {
            for (int i = 0; (i < prefetch_blk_)
                    && (prefetches_issued_ < nb_cache_lines_to_prefetch_);
                    i++, prefetches_issued_++) {
                prefetch_inst_(cg_->EVEX_compress_addr(
                        reg_base_addr_, (cache_block_size_ * prefetch_distance_)
                                        * sizeof(data_t)
                                + (prefetches_issued_ * 64)));
            }
        }
    }

private:
    void prefetch_inst_(const Xbyak::Address &addr)
    {
        switch (cache_type_) {
        case L1: cg_->prefetcht0(addr); break;
        case L2: cg_->prefetcht1(addr); break;
        case L3: cg_->prefetcht2(addr); break;
        default:
            break; // TODO: raise an exception or put an assert
        }
    }

    jit_generator *cg_;
    Xbyak::Reg64 reg_base_addr_;
    cache_t cache_type_;
    size_t cache_block_size_ = 0;
    size_t nb_cache_lines_to_prefetch_ = 0;
    int prefetches_issued_ = 0;
    int prefetch_spread_ = 0;
    int prefetch_blk_ = 0;
    int prefetch_distance_ = 0;
};

void _jit_avx512_common_conv_winograd_data_kernel_f32::gemm_loop_generate(
        bool is_beta_zero)
{
    // const int dimK_simd_block = jcp.dimK_reg_block;

    // for (int dimM_block =0; dimM_block < jcp.dimM_block; dimM_block++)
    //     for (int dimK_block = 0; dimK_block < jcp.dimK_block; dimK_block++)
    //         for (int dimK_reg_block= 0; dimK_reg_block < jcp.dimK_reg_block;
    //         dimK_reg_block++)
    //                 for (int tile =0; tile < jcp.dimN_reg_block; tile++)
    //                     C[dimM_block][tile] +=
    //                     A[dimM_block][dimK_block][dimK_reg_block] *
    //                     broadcast(B[dimK_block][tile][dimK_reg_block]);
    // 1) We do register blocking on A[dimM_block][dimK_block][dimK_reg_block],
    // so we load it before the loop on tile
    // 2) the loop on tile must be fully unrolled. Don't know about the one on
    // dimK_reg_block. I think it should be

    auto inner_loops = [=]() {
        Label dimM_block_loop, dimK_block_loop;
        const int inc_dimK_reg_block = jcp.ver == ver_4fma ? 4 : 1;
        const int fma_ipc = jcp.ver == ver_4fma ? 1 : 2;

        prefetcher_t<float> L1_pf(this, reg_srcB, L1,
                jcp.dimN_reg_block * jcp.dimK_reg_block,
                jcp.dimK_reg_block * jcp.dimN_reg_block / inc_dimK_reg_block,
                fma_ipc);
        prefetcher_t<float> L2_pf(this, reg_srcB, L2,
                jcp.dimN_reg_block * jcp.dimK_reg_block,
                jcp.dimK_reg_block * jcp.dimN_reg_block / inc_dimK_reg_block,
                fma_ipc);

        if (jcp.dimM_block > 1) {
            mov(reg_dimM_block_loop_cnt, jcp.dimM_block);
            L(dimM_block_loop);
        }
        {
            // First, we zero the accumulators if first nb_ic iteration,
            // otherwise we load them
            for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                Zmm zmm(jcp.zmm_start + tile);
                if (is_beta_zero)
                    vpxord(zmm, zmm, zmm);
                else
                    vmovups(zmm, zword[reg_dstC + 64 * tile]);
            }

            if (jcp.dimK_block > 1) {
                mov(reg_dimK_block_loop_cnt, jcp.dimK_block);
                L(dimK_block_loop);
            }
            {
                auto load_A = [=](int reg_idx, int offset) {
                    for (int i = 0; i < inc_dimK_reg_block; i++)
                        vmovups(Zmm(reg_idx + i),
                                zword[reg_srcA + 64 * (offset + i)]);
                };

                // Used when doing double buffering
                int next = 0;
                if (jcp.double_buffering) {
                    load_A(next, 0);
                }
                for (int dimK_reg_block = 0;
                        dimK_reg_block < jcp.dimK_reg_block;
                        dimK_reg_block += inc_dimK_reg_block) {
                    int current;
                    /* Loading the next vector from A */
                    current = next;
                    if (jcp.double_buffering) {
                        next = (dimK_reg_block + inc_dimK_reg_block)
                                % (2 * inc_dimK_reg_block);
                        load_A(next, dimK_reg_block + inc_dimK_reg_block);
                    } else {
                        next = 0;
                        load_A(next, dimK_reg_block);
                    }
                    /* Performing the fmas */
                    for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                        Zmm zmm(jcp.zmm_start + tile);
#if !defined(SKX_OPT)
                        // TODO: prefetch on SKX -wxy
                        L1_pf.prefetch(
                                dimK_reg_block * jcp.dimN_reg_block + tile);
#endif
                        if (jcp.ver == ver_4fma)
                            v4fmaddps(zmm, Zmm(current),
                                    EVEX_compress_addr(reg_srcB,
                                              64 * tile + dimK_reg_block * 4));
                        else
                            vfmadd231ps(zmm, Zmm(current),
                                    EVEX_compress_addr(reg_srcB,
                                                64 * tile + dimK_reg_block * 4,
                                                true));
#if !defined(SKX_OPT)
                        // TODO: prefetch on SKX -wxy
                        L2_pf.prefetch(
                                dimK_reg_block * jcp.dimN_reg_block + tile);
#endif
                    }
                }

                // Fix a bug when oc_block > 1, -wxy
                add(reg_srcA, jcp.dimK_reg_block * 64);
                add(reg_srcB, jcp.dimN_reg_block * 64);
                if (jcp.dimK_block > 1) {
                    sub(reg_dimK_block_loop_cnt, 1);
                    jnz(dimK_block_loop);
                }
            }

            // We write the results in destination
            for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                Zmm zmm(jcp.zmm_start + tile);
                // In W_SGD or W_S_GD, output will be reused. -wxy
                if (jcp.dimK_nb_block == 1
                        && (jcp.sched_policy == WSCHED_DATA_W_S_G_D
                            || jcp.sched_policy == WSCHED_DATA_W_SGit_D))
                    vmovntps(zword[reg_dstC + 64 * tile], zmm);
                else
                    vmovups(zword[reg_dstC + 64 * tile], zmm);
            }

            if (jcp.dimM_block > 1) {
                sub(reg_srcB, jcp.dimK_block * jcp.dimN_reg_block * 64);
                add(reg_dstC, jcp.dimN_reg_block * 64);
                sub(reg_dimM_block_loop_cnt, 1);
                jnz(dimM_block_loop);
            }
        }
    };

    /* Preamble */
    // register used to handle long fma encoding
    push(reg_EVEX_max_8b_offt);
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);

    /* kernel */
    inner_loops();

    /* Postamble */
    pop(reg_EVEX_max_8b_offt);
    ret();
}

status_t _jit_avx512_common_conv_winograd_data_kernel_f32::init_conf_common(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d)
{

    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = 16;

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
    jcp.r_pad = nstl::max(
            0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);
    jcp.b_pad = nstl::max(
            0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih - jcp.t_pad);
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

    // Winograd specific initialization
    const int tile_size = jcp.alpha - 2;
    /* Assumption: padding = 1*/
    jcp.itiles = (jcp.ow + tile_size - 1) / tile_size;
    jcp.jtiles = (jcp.oh + tile_size - 1) / tile_size;
    jcp.ntiles = jcp.mb * jcp.itiles * jcp.itiles;

    // Checking conditions not supported by these kernels
    if (jcp.ngroups != 1)
        return status::unimplemented;
    if ((jcp.kh != 3) || (jcp.kw != 3))
        return status::unimplemented;
    if ((jcp.stride_h != 1) || (jcp.stride_w != 1))
        return status::unimplemented;
    if ((jcp.ic % simd_w) != 0 || (jcp.oc % simd_w) != 0)
        return status::unimplemented;

    if (src_d.format() != nChw16c)
        return status::unimplemented;
    if (weights_d.format() != (with_groups ? gOIhw16i16o : OIhw16i16o))
        return status::unimplemented;
    if (dst_d.format() != nChw16c)
        return status::unimplemented;

    jcp.ver = mayiuse(avx512_mic_4ops) ? ver_4fma : ver_fma;

    return status::success;
}

bool check_cond1(int dimN_reg_block, int dimK_block, int dimK_reg_block,
        int dimM_block, int dimM_simd_block, float C)
{
    float lhs = (dimM_block * dimN_reg_block * dimM_simd_block
                        + dimM_block * dimK_block * dimK_reg_block
                                * dimM_simd_block
                        + dimK_block * dimN_reg_block * dimK_reg_block)
            * sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs < rhs);
}

bool check_cond1_bis(int dimN_reg_block, int dimK_block, int dimK_reg_block,
        int dimM_block, int dimM_simd_block, float C)
{
    float lhs = (dimM_block * dimK_block * dimK_reg_block * dimM_simd_block
                        + dimK_block * dimN_reg_block * dimK_reg_block)
            * sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs < rhs);
}

bool check_cond2(int nb_dimN_reg_block, int dimN_reg_block, int dimK_nb_block,
        int dimK_block, int dimK_reg_block, int dimM_block, int dimM_simd_block,
        float C)
{
    int lhs = (nb_dimN_reg_block * dimM_block * dimN_reg_block * dimM_simd_block
                      + dimK_nb_block * dimM_block * dimK_block * dimK_reg_block
                              * dimM_simd_block
                      + nb_dimN_reg_block * dimK_nb_block * dimK_block
                              * dimN_reg_block * dimK_reg_block)
            * sizeof(float);
    float rhs = C * L2_cache_size;
    return (lhs < rhs);
}

status_t set_wsched_DATA_W_SGDt(jit_conv_winograd_conf_t &jcp)
{
    /*
       WSCHED_DATA_W_SGDt
       ============

       Intuition:
       If N is pretty much big while K/M is relatively small, we could split
       N into multiple tile blocks and group each tile block's src-transform,
       gemm and dst-transform into one thread for better L2 cache locality.

       Parameter selection: 
       1. V:thread-size + M:thread-size:  [C1, C2] * L2_cache_size
       2. work-amount is within (T1 ~ T2) * OMP-MAX-THREADS (thread balance)
       3. N:reg-block as big as possible
       4. V:K-block-size + M:M-block-size < C * L1_cache_size
       
       Steps:
       Search start:
         dimN_reg_block = max-possible
         dimN_block = 1
       check-condition:
         size-two-small: dimN_block UP
         size-two-big: dimN_reg_block DOWN

       */
    const float C1_1 = 0.01, C1_2 = 0.8;
    const float C2_1 = 0.2, C2_2 = 1.1; // overflow to L3
    const int T1 = 3, T2 = 256;
    const int min_dimN_reg_block = 7;

    int dimN_reg_block = jcp.dimN, current_dimN_reg_block = jcp.dimN;
    int dimN_nb_block = jcp.dimN, current_dimN_nb_block = jcp.dimN;
    auto test_cond_dimN_reg_block = [](jit_conv_winograd_conf_t jcp,
            int dimN_reg_block, int current_best) {
        return (dimN_reg_block >= min_dimN_reg_block)
                && (dimN_reg_block <= jcp.nb_reg)
                && (dimN_reg_block < current_best);
    };
    auto test_cond_dimN_nb_block = [](jit_conv_winograd_conf_t jcp,
            int dimN_nb_block, int current_best) {
        int V_thread_size = jcp.dimK * (jcp.dimN / dimN_nb_block)
            * jcp.alpha * jcp.alpha * sizeof(float);
        int M_thread_size = jcp.dimM * (jcp.dimN / dimN_nb_block)
            * jcp.alpha * jcp.alpha * sizeof(float);
        return ((V_thread_size + M_thread_size >= C2_1 * L2_cache_size)
                && (V_thread_size + M_thread_size <= C2_2 * L2_cache_size)
                && dimN_nb_block >= T1 * omp_get_max_threads()
                && dimN_nb_block <= T2 * omp_get_max_threads());
    };
    auto test_cond_dimK_block = [](jit_conv_winograd_conf_t jcp,
            int dimK_block, int current_best) {
        float ratio = jcp.dimK * 1. / (jcp.dimK + jcp.dimM);
        int V_block_size = jcp.dimK_reg_block * jcp.dimN_reg_block
            * dimK_block * sizeof(float);

        return ((V_block_size >= C1_1 * ratio * L1_cache_size)
                && (V_block_size <= C1_2 * ratio * L1_cache_size));
    };
    auto test_cond_dimM_block = [](jit_conv_winograd_conf_t jcp,
            int dimM_block, int current_best) {
        float ratio = jcp.dimM * 1. / (jcp.dimK + jcp.dimM);
        int M_block_size = jcp.dimM_simd_block * jcp.dimN_reg_block
            * dimM_block * sizeof(float);

        return ((M_block_size >= C1_1 * ratio * L1_cache_size)
                && (M_block_size <= C1_2 * ratio * L1_cache_size));
    };

    bool meet_N_blocking = false;
    for (int i = 0; i < 3; i++) {
        dimN_reg_block = get_max_divisor_satisfying_cond(
                jcp, jcp.dimN, current_dimN_reg_block, test_cond_dimN_reg_block);
        if (dimN_reg_block == current_dimN_reg_block) // fail
            break;
        dimN_nb_block = get_max_divisor_satisfying_cond(jcp,
                jcp.dimN / dimN_reg_block, current_dimN_nb_block,
                test_cond_dimN_nb_block);
        if (dimN_nb_block == current_dimN_nb_block) { // fail
            current_dimN_nb_block = dimN_nb_block;
            current_dimN_reg_block = dimN_reg_block;
        } else {
            meet_N_blocking = true;
            break;
        }
    }

    if (!meet_N_blocking)
        return status::unimplemented;

    jcp.dimN_reg_block = dimN_reg_block;
    jcp.dimN_nb_block = dimN_nb_block;
    jcp.dimN_block = jcp.dimN / jcp.dimN_reg_block / jcp.dimN_nb_block;

    printf("V_thread_size=%ld\n",
            jcp.ic * (jcp.dimN / jcp.dimN_nb_block)
            * jcp.alpha * jcp.alpha * sizeof(float));
    printf("M_thread_size=%ld\n",
            jcp.oc * (jcp.dimN / jcp.dimN_nb_block)
            * jcp.alpha * jcp.alpha * sizeof(float));

    jcp.dimK_block = get_max_divisor_satisfying_cond(jcp,
            jcp.dimK / jcp.dimK_reg_block,
            1, test_cond_dimK_block);
    //jcp.dimK_block = 2;
    jcp.dimK_nb_block = jcp.dimK / jcp.dimK_block / jcp.dimK_reg_block;

    jcp.dimM_block = get_max_divisor_satisfying_cond(jcp,
            jcp.dimM / jcp.dimM_simd_block,
            1, test_cond_dimM_block);
    //jcp.dimM_block = 2;
    jcp.dimM_nb_block = jcp.dimM / jcp.dimM_block / jcp.dimM_simd_block;

    jcp.sched_policy = WSCHED_DATA_W_SGDt;
    printf("check DATA_W_SGDt\n");

    return status::success;
}

status_t set_wsched_DATA_W_S_GDot(jit_conv_winograd_conf_t &jcp)
{
    /*
       WSCHED_DATA_W_S_GDot
       ============

       Intuition:
       If N is not big enough to feed the number of threads/cores while M is
       relatively big (compared to K), we could split N * M into multiple
       tile blocks and group each tile block's gemm and dst-transform into
       one thread for better L2 cache locality.
      
       */


    return status::unimplemented;
    //jcp.sched_policy = WSCHED_DATA_W_S_GDot;
    //printf("check DATA_W_S_GDot\n");
    //return status::success;
}

status_t set_wsched_DATA_W_SGit_D(jit_conv_winograd_conf_t &jcp)
{
    /*
       WSCHED_DATA_W_SGit_D
       ============

       Intuition:
       If N is not big enough to feed the number of threads/cores while K is
       relatively big (compared to M), we could split N * K into multiple
       tile blocks and group each tile block's src-transform and gemm into
       one thread for better L2 cache locality.
      
       */


    return status::unimplemented;
    //jcp.sched_policy = WSCHED_DATA_W_SGit_D;
    //printf("check DATA_W_SGit_D\n");
    //return status::success;
}

status_t set_wsched_DATA_W_S_G_D(jit_conv_winograd_conf_t &jcp)
{
    /*
     Parameter selection: WSCHED_DATA_W_S_G_D 

     [1] L1_cache condition with stores
     [dimM_block][dimN_reg_block][simd_w]
     + [dimM_block][dimK_block][simd_w][simd_w]
     + [dimK_block][dimN_reg_block][simd_w] < C * L1_cache_size

     [1bis] L1_cache condition with non-temporal stores
     + [dimM_block][dimK_block][simd_w][simd_w]
     + [dimK_block][dimN_reg_block][simd_w] < C * L1_cache_size

     [2] L2 cache condition
     [nb_dimN_reg_block][dimM_block][dimN_reg_block][simd_w]
     + [dimK_nb_block][dimM_block][dimK_block][simd_w][simd_w]
     + [nb_dimN_reg_block][dimK_nb_block][dimK_block][dimN_reg_block][simd_w]
     < C * L2_cache_size

     with C~1/2. C is here to prevent the HW prefetcher from evicting the needed
     data.

     0) pick dimN_reg_block such that it is just above what is needed
     ot cover load latencies. Ideally, it should be bigger than 14 to
     hide latencies.

     1) pick dimK_block. Assuming jcp.dimM_block=1, Check if [1bis]
     holds with dimK_block = dimK / simd_w.

       i.  If it does, (with the current POR topologies, it should
       almost always be the case)

         a. pick dimK_block=dimK/simd_w as it enables to stream the
         output of the GEMM and save some bandwidth for reading and
         will save some work for the prefetcher.

         b. pick dimM_block as big as possible such that:
           - [1bis] holds

         c. pick nb_dimN_reg_block such that [2] holds

       ii. If it does not, it is not clear what is the best tradeoff:
          a. maximize dimK_block to minimize the number of writes?
          b. maximize dimM_block to maximize L1 cache reuse
          c. find the best (dimM_block, dimK_block couple) depending on
            the number of use-per-load and write instructions? What
            would be the weight to put to each?
          For now, we will stick with a., but it is worth
          investigating c. Note that this will come with some
          maintenance burden as the weights will likely be
          architecture dependant
   */
 
    //******************* Choosing dimN_reg_block *******************//
    // Fix issue when N (ex. mb1ih28oh28) is small. And choose a bigger
    // dimN_reg_block  -wxy
#define MIN_REQUIRED_DIMN_REG_BLOCK 1
    auto test_cond_dimN_reg_block = [](jit_conv_winograd_conf_t jcp,
            int dimN_reg_block, int current_best) {
        return (dimN_reg_block >= MIN_REQUIRED_DIMN_REG_BLOCK)
            && (dimN_reg_block <= jcp.nb_reg)
            && (dimN_reg_block < current_best);
    };
    jcp.dimN_reg_block = get_max_divisor_satisfying_cond(
            jcp, jcp.dimN, jcp.dimN, test_cond_dimN_reg_block);
    //jcp.dimN_reg_block = 7; //wxy

    //********************* Choosing dimK_block **********************//
    auto test_cond1_dimK_block = [](
            jit_conv_winograd_conf_t jcp, int dimK_block, int current_best) {
        return check_cond1(jcp.dimN_reg_block, dimK_block, jcp.dimK_reg_block,
                1, jcp.dimM_simd_block, .75f)
            && (dimK_block > current_best);
    };

    auto test_cond1_bis_dimK_block = [](
            jit_conv_winograd_conf_t jcp, int dimK_block, int current_best) {
        return check_cond1_bis(jcp.dimN_reg_block, dimK_block,
                jcp.dimK_reg_block, 1, jcp.dimM_simd_block, .9f)
            && (dimK_block > current_best);
    };

    jcp.dimK_block = get_divisor_satisfying_cond(
            jcp, jcp.dimK / jcp.dimK_reg_block, 1, test_cond1_bis_dimK_block);
    // If we are not able to use streams, we fall back to condition [1]
    if (jcp.dimK_block < jcp.dimK / jcp.dimK_reg_block)
        jcp.dimK_block = get_divisor_satisfying_cond(
                jcp, jcp.dimK / jcp.dimK_reg_block, 1, test_cond1_dimK_block);
    //jcp.dimK_block = 2; //wxy
    jcp.dimK_nb_block = (jcp.dimK / jcp.dimK_reg_block) / jcp.dimK_block;

    //********************* Choosing dimM_block **********************//
    //jcp.dimM_simd_block = 16;
    /*XXX: Why C=0.5 here but C=0.75 for dimK_block?*/
    auto test_cond1_dimM_block = [](
            jit_conv_winograd_conf_t jcp, int dimM_block, int current_best) {
        return check_cond1(jcp.dimN_reg_block, jcp.dimK_block,
                jcp.dimK_reg_block, dimM_block, jcp.dimM_simd_block, .5f)
            && (dimM_block > current_best);
    };

    auto test_cond1_bis_dimM_block = [](
            jit_conv_winograd_conf_t jcp, int dimM_block, int current_best) {
        return check_cond1_bis(jcp.dimN_reg_block, jcp.dimK_block,
                jcp.dimK_reg_block, dimM_block, jcp.dimM_simd_block, .3f)
            && (dimM_block > current_best);
    };

    if (jcp.dimK_block < jcp.dimK / jcp.dimK_reg_block)
        jcp.dimM_block = get_divisor_satisfying_cond(
                jcp, jcp.dimM / jcp.dimM_simd_block, 1, test_cond1_dimM_block);
    else
        jcp.dimM_block = get_divisor_satisfying_cond(jcp,
                jcp.dimM / jcp.dimM_simd_block, 1, test_cond1_bis_dimM_block);
    //jcp.dimM_block = 2; //wxy
    jcp.dimM_nb_block = (jcp.dimM / jcp.dimM_simd_block) / jcp.dimM_block;

    //******************* Choosing dimN_block *******************//
    auto test_cond2_dimN_block = [](
            jit_conv_winograd_conf_t jcp, int dimN_block, int current_best) {
        return check_cond2(dimN_block, jcp.dimN_reg_block, jcp.dimK_nb_block,
                jcp.dimK_block, jcp.dimK_reg_block, jcp.dimM_block,
                jcp.dimM_simd_block, .6f)
            && (dimN_block > current_best);
    };

    jcp.dimN_block = get_divisor_satisfying_cond(
            jcp, jcp.dimN / jcp.dimN_reg_block, 1, test_cond2_dimN_block);
    //jcp.dimN_block = 16; //wxy
    jcp.dimN_nb_block = jcp.dimN / (jcp.dimN_reg_block * jcp.dimN_block);

    jcp.sched_policy = WSCHED_DATA_W_S_G_D;
    printf("check DATA_W_S_G_D\n");

    return status::success;
}

status_t _jit_avx512_common_conv_winograd_data_kernel_f32::init_conf_kernel(
        jit_conv_winograd_conf_t &jcp, int dimM, int dimN, int dimK)
{
    jcp.dimK_reg_block = 16;
    jcp.dimM_simd_block = 16;

    // TODO: replace double buffering with nuple buffering to maximize register
    // usage.
    // the choice of the number of buffers will then come after choosing
    // dimN_reg_block
    // Do we do double buffering?
    jcp.double_buffering = true;
    if (jcp.double_buffering)
        jcp.zmm_start = 2 * ((jcp.ver == ver_4fma) ? 4 : 2);
    else
        jcp.zmm_start = 1;
    jcp.nb_reg = 32 - jcp.zmm_start;

    jcp.dimN = dimN;
    jcp.dimK = dimK;
    jcp.dimM = dimM;

    // For bwd-weight compatibility
    jcp.tile_4fma = 1;

    jcp.sched_policy = WSCHED_INVALID;
    status_t res;
    if ((res = set_wsched_DATA_W_SGDt(jcp))   == status::success ||
        (res = set_wsched_DATA_W_S_GDot(jcp))  == status::success ||
        (res = set_wsched_DATA_W_SGit_D(jcp))  == status::success ||
        (res = set_wsched_DATA_W_S_G_D(jcp)) == status::success)
        ;

    return res;
}

status_t jit_avx512_common_conv_winograd_fwd_kernel_f32::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, bool with_relu,
        double relu_negative_slope)
{
    status_t st = init_conf_common(jcp, cd, src_d, weights_d, dst_d);

    if (st != status::success)
        return st;

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    status_t res = init_conf_kernel(jcp, jcp.oc, jcp.ntiles, jcp.ic);
    jcp.ic_simd_block = jcp.dimK_reg_block;
    jcp.ic_block = jcp.dimK_block;
    jcp.nb_ic = jcp.dimK_nb_block;

    jcp.oc_simd_block = jcp.dimM_simd_block;
    jcp.oc_block = jcp.dimM_block;
    jcp.nb_oc = jcp.dimM_nb_block;

    jcp.tile_block_ur = jcp.dimN_reg_block;
    jcp.nb_tile_block_ur = jcp.dimN_block;
    jcp.tile_block = jcp.dimN_nb_block;
    jcp.tile_4fma_padding = 0; // only relevant for backward weights

    printf("ic_simd_block=%d, ic_block=%d, nb_ic=%d\n",
            jcp.ic_simd_block, jcp.ic_block, jcp.nb_ic);
    printf("oc_simd_block=%d, oc_block=%d, nb_oc=%d\n",
            jcp.oc_simd_block, jcp.oc_block, jcp.nb_oc);
    printf("tile_block_ur=%d, nb_tile_block_ur=%d, tile_block=%d\n",
            jcp.tile_block_ur, jcp.nb_tile_block_ur, jcp.tile_block);
    return res;
}

status_t jit_avx512_common_conv_winograd_bwd_data_kernel_f32::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    status_t st = init_conf_common(jcp, cd, diff_src_d, weights_d, diff_dst_d);

    if (st != status::success)
        return st;

    status_t res = init_conf_kernel(jcp, jcp.ic, jcp.ntiles, jcp.oc);
    jcp.oc_simd_block = jcp.dimK_reg_block;
    jcp.oc_block = jcp.dimK_block;
    jcp.nb_oc = jcp.dimK_nb_block;
    jcp.ic_simd_block = jcp.dimM_simd_block;
    jcp.ic_block = jcp.dimM_block;
    jcp.nb_ic = jcp.dimM_nb_block;
    jcp.tile_block_ur = jcp.dimN_reg_block;
    jcp.nb_tile_block_ur = jcp.dimN_block;
    jcp.tile_block = jcp.dimN_nb_block;
    jcp.tile_4fma_padding = 0; // only relevant for backward weights

    return res;
}
void jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::transpose_ker_generate()
{
    auto load_B = [=](int reg_idx, int offset) {
        for (int i = 0; i < 4; i++) {
            vmovups(Zmm(reg_idx + i), zword[reg_origB + (offset + i) * jcp.dimN_reg_block * sizeof(float)]);
        }
    };

    int curr = 0;
    for (int j = 0; j < jcp.alpha; j++) {
        for (int i = 0; i < jcp.alpha; i++) {
            int origB_offset = (j * jcp.alpha + i) * jcp.dimK_4fma;
            int transB_offset = (j * jcp.alpha + i) * jcp.dimK_nb_block *
                jcp.dimN_block * jcp.dimK_block * jcp.dimK_reg_block *
                jcp.dimK_4fma * jcp.dimN_reg_block;
            for (int tb = 0; tb < jcp.dimK_4fma; tb+=4) {
                /*double buffering to hide load latencies*/
                int next = (curr + 4) % 8;
                if (i == 0 && tb == 0) {
                    load_B(0, origB_offset);
                }
                if (tb + 4 < (jcp.dimK_4fma -1)) {
                    load_B(next, origB_offset + 4);
                } else if (i < jcp.alpha - 1) {
                    load_B(next, origB_offset + jcp.dimK_4fma);
                }

                vunpcklps(Zmm(8), Zmm(curr), Zmm(curr + 1));
                vunpcklps(Zmm(9), Zmm(curr + 2), Zmm(curr + 3));
                vunpckhps(Zmm(curr), Zmm(curr), Zmm(curr + 1));
                vunpckhps(Zmm(curr + 1), Zmm(curr + 2), Zmm(curr + 3));

                vunpcklpd(Zmm(curr + 2), Zmm(8), Zmm(9));
                vunpckhpd(Zmm(curr + 3), Zmm(8), Zmm(9));

                vunpcklpd(Zmm(8), Zmm(curr), Zmm(curr + 1));
                vunpckhpd(Zmm(9), Zmm(curr), Zmm(curr + 1));

                vmovntps(zword[reg_transB
                        + sizeof(float) * (transB_offset + tb * jcp.dimN_reg_block)],
                        Zmm(curr+2));
                vmovntps(zword[reg_transB
                        + sizeof(float) * (transB_offset + (tb + 1) * jcp.dimN_reg_block)],
                        Zmm(curr+3));
                vmovntps(zword[reg_transB
                        + sizeof(float) * (transB_offset + (tb + 2) * jcp.dimN_reg_block)],
                        Zmm(8));
                vmovntps(zword[reg_transB
                        + sizeof(float) * (transB_offset + (tb + 3) * jcp.dimN_reg_block)],
                        Zmm(9));
                curr = next;

            }
        }
    }
    ret();
}
void jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::gemm_loop_generate(
        bool is_first_tile)
{
    // for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++)
    //     for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++)
    //             for (int nb_tile_block_ur = 0; nb_tile_block_ur <
    //             jcp.nb_tile_block_ur; nb_tile_block_ur++)
    //                 for (int tile_block_ur = 0; tile_block_ur <
    //                 jcp.tile_block_ur; tile_block_ur++)
    //                     for (int ifm3 = 0; ifm3 < jcp.ic_reg_block; ++ifm3)
    //                         U[ofm2][ifm2][ofm3][ifm3][0:oc_simd_block] +=
    //                             M[ofm2][ofm3][nb_tile_block_ur][tile_block_ur][0:oc_simd_block]
    //                              *
    //                              broadcast(V[ifm2][nb_tile_block_ur][ifm3][tile_block_ur])
    auto inner_loops = [=]() {
        int inc_fma = jcp.ver == ver_4fma ? 4 : 1;
        const int fma_ipc = jcp.ver == ver_4fma ? 1 : 2;
        prefetcher_t<float> L1_pf(this, reg_srcB, L1,
                jcp.dimK_reg_block * jcp.dimN_reg_block * jcp.dimK_4fma,
                jcp.dimK_reg_block * jcp.dimN_reg_block * jcp.dimK_4fma
                        / inc_fma,
                fma_ipc);
        prefetcher_t<float> L2_pf(this, reg_srcB, L2,
                jcp.dimK_reg_block * jcp.dimN_reg_block * jcp.dimK_4fma,
                jcp.dimK_reg_block * jcp.dimN_reg_block * jcp.dimK_4fma
                        / inc_fma,
                fma_ipc);

        auto load_A = [=](int reg_idx, int offset) {
            for (int i = 0; i < inc_fma; i++) {
                vmovups(Zmm(reg_idx + i),
                        zword[reg_srcA +
                        sizeof(float) * jcp.dimM_simd_block * (offset + i)]);
            }
        };

        Label dimM_block_loop, dimK_block_loop, dimN_block_loop;
        if (jcp.dimM_block > 1) {
            mov(reg_dimM_block_loop_cnt, jcp.dimM_block);
            L(dimM_block_loop);
        }
        { /************* OC_block (M) loop ***********/
            if (jcp.dimN_block > 1) {
                mov(reg_dimN_block_loop_cnt, jcp.dimN_block);
                L(dimN_block_loop);
            }
            { /*************** IC_block (N) loop *********/
                for (int dimN_reg_block = 0;
                        dimN_reg_block < jcp.dimN_reg_block; ++dimN_reg_block) {
                    Zmm zmm(jcp.zmm_start + dimN_reg_block);
                    if (is_first_tile)
                        vpxord(zmm, zmm, zmm);
                    else
                        vmovups(zmm, zword[reg_dstC +
                                dimN_reg_block * jcp.dimM_simd_block *
                                sizeof(float)]);
                }

                if (jcp.dimK_block > 1) {
                    mov(reg_dimK_block_loop_cnt, jcp.dimK_block);
                    L(dimK_block_loop);
                }
                { /************* nb_tile_ur(K) loop ********/
                    int next = 0;
                    if (jcp.double_buffering) {
                        load_A(next, 0);
                    }
                    for (int dimK_reg_block = 0;
                            dimK_reg_block < jcp.dimK_reg_block;
                            dimK_reg_block++) {
                        int srcB_offset = dimK_reg_block * jcp.dimK_4fma
                                * jcp.dimN_reg_block;
                        for (int dimK_4fma = 0; dimK_4fma < jcp.dimK_4fma;
                                dimK_4fma += inc_fma) {
                            int current = next;
                            if (jcp.double_buffering) {
                                next = (dimK_reg_block * jcp.dimK_4fma
                                               + dimK_4fma + inc_fma)
                                        % (2 * inc_fma);
                                load_A(next, dimK_reg_block * jcp.dimK_4fma
                                                + dimK_4fma + inc_fma);
                            } else {
                                next = 0;
                                load_A(next, dimK_reg_block * jcp.dimK_4fma
                                                + dimK_4fma);
                            }
                            for (int dimN_reg_block = 0;
                                    dimN_reg_block < jcp.dimN_reg_block;
                                    ++dimN_reg_block) {
#if !defined(SKX_OPT)
                                L1_pf.prefetch(srcB_offset / inc_fma
                                        + dimK_4fma / inc_fma
                                                * jcp.dimN_reg_block
                                        + dimN_reg_block);
                                L2_pf.prefetch(srcB_offset / inc_fma
                                        + dimK_4fma / inc_fma
                                                * jcp.dimN_reg_block
                                        + dimN_reg_block);
#endif
                                if (jcp.ver == ver_4fma) {
                                    int srcB_trans_offset = (dimK_4fma / 4) * 64
                                            + dimK_4fma % 4;
                                    v4fmaddps(
                                            Zmm(jcp.zmm_start + dimN_reg_block),
                                            Zmm(current),
                                            EVEX_compress_addr(reg_srcB,
                                                    sizeof(float) * (
                                                        srcB_offset +
                                                        srcB_trans_offset +
                                                        (dimN_reg_block % 4) * 16 +
                                                        (dimN_reg_block / 4) * 4)));
                                } else {
                                    vfmadd231ps(
                                            Zmm(jcp.zmm_start + dimN_reg_block),
                                            Zmm(current),
                                            EVEX_compress_addr(reg_srcB,
                                                sizeof(float) * (srcB_offset + dimN_reg_block),
                                                    true));
                                }
                            }
                        }
                    }
                }

                add(reg_srcA, jcp.dimK_reg_block * jcp.dimK_4fma
                                * jcp.dimM_simd_block * sizeof(float));
                add(reg_srcB, jcp.dimK_reg_block * jcp.dimN_reg_block
                                * jcp.dimK_4fma * sizeof(float));
                if (jcp.dimK_block > 1) {
                    sub(reg_dimK_block_loop_cnt, 1);
                    jnz(dimK_block_loop);
                }

                /******** Write C back to memory *******/
                for (int dimN_reg_block = 0;
                        dimN_reg_block < jcp.dimN_reg_block; ++dimN_reg_block) {
                    Zmm zmm(jcp.zmm_start + dimN_reg_block);
                    //vmovntps(zword[reg_dstC +
                    vmovups(zword[reg_dstC +
                            dimN_reg_block * jcp.dimM_simd_block * sizeof(float)],
                            zmm);
                }

                sub(reg_srcA, jcp.dimK_block * jcp.dimK_reg_block *
                        jcp.dimK_4fma * jcp.dimM_simd_block * sizeof(float));
                add(reg_dstC, jcp.dimN_reg_block * jcp.dimM_simd_block
                        * sizeof(float));
                if (jcp.dimN_block > 1) {
                    sub(reg_dimN_block_loop_cnt, 1);
                    jnz(dimN_block_loop);
                }
            }

            if (jcp.dimM_block > 1) {
                sub(reg_srcB, jcp.dimN_block * jcp.dimK_block
                                * jcp.dimK_reg_block * jcp.dimN_reg_block
                                * jcp.dimK_4fma * sizeof(float));
                add(reg_srcA, jcp.dimK_block * jcp.dimK_reg_block
                                * jcp.dimK_4fma * jcp.dimM_simd_block * sizeof(float));
                sub(reg_dimM_block_loop_cnt, 1);
                jnz(dimM_block_loop);
            }
        }
    };

    /* Preamble */
    // register used to handle long fma encoding
    push(reg_EVEX_max_8b_offt);
    push(reg_dimK_block_loop_cnt);
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
    mov(reg_srcA, reg_srcA_const);
    inner_loops();

    /* Postamble */
    pop(reg_dimK_block_loop_cnt);
    pop(reg_EVEX_max_8b_offt);
    ret();
}

bool check_cond1_wu(int dimM_block, int dimM_simdw, int dimK_block,
        int dimK_reg_block, int dimK_4fma, int dimN_reg_block, float C)
{
    float lhs = dimM_block * dimN_reg_block * dimM_simdw;
    lhs += dimM_block * dimK_block * dimK_reg_block * dimK_4fma * dimM_simdw;
    lhs += dimK_block * dimN_reg_block * dimK_reg_block * dimK_4fma;
    lhs *= sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs <= rhs);
}

bool check_cond1bis_wu(int dimM_block, int dimM_simdw, int dimK_block,
        int dimK_reg_block, int dimK_4fma, int dimN_reg_block, float C)
{
    float lhs
            = dimM_block * dimK_block * dimK_reg_block * dimK_4fma * dimM_simdw;
    lhs += dimK_block * dimN_reg_block * dimK_reg_block * dimK_4fma;
    lhs *= sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs <= rhs);
}

bool check_cond2bis_wu(int dimM_block, int dimM_simdw, int dimK_block,
        int dimK_reg_block, int dimK_4fma, int dimN_block, int dimN_reg_block,
        float C)
{
    float lhs
            = dimM_block * dimM_simdw * dimK_block * dimK_reg_block * dimK_4fma;
    lhs += dimK_block * dimK_reg_block * dimK_4fma * dimN_block
            * dimN_reg_block;
    lhs *= sizeof(float);
    float rhs = C * L2_cache_size;
    return (lhs <= rhs);
}
bool check_cond2_wu(int dimM_block, int dimM_simdw, int dimK_block,
        int dimK_reg_block, int dimK_4fma, int dimN_block, int dimN_reg_block,
        float C)
{
    float lhs = dimM_block * dimM_simdw * dimN_block * dimN_reg_block;
    lhs += dimM_block * dimM_simdw * dimK_block * dimK_reg_block * dimK_4fma;
    lhs += dimK_block * dimK_reg_block * dimK_4fma * dimN_block
            * dimN_reg_block;
    lhs *= sizeof(float);
    float rhs = C * L2_cache_size;
    return (lhs <= rhs);
}

status_t set_wsched_WEI_SDGt_W(jit_conv_winograd_conf_t &jcp)
{
    /*
       Parameter selection: 
       1. V:thread-size + M:thread-size + U:size: [C1, C2] * L2_cache_size
       2. work-amount is within (T1 ~ T2) * OMP-MAX-THREADS (thread balance)
       3. V:N-block-size + M:M-block-size + U:M-block-size < C * L1_cache_size

       */ 

    const float C1_1 = 0.01, C1_2 = 0.9; // L1
    const float C2_1 = 0.2, C2_2 = 1.1; // L2, overflow to L3
    const int T1 = 3, T2 = 256;

    int dimK_reg_block = jcp.dimK, current_dimK_reg_block = jcp.dimK;
    int dimK_nb_block = jcp.dimK, current_dimK_nb_block = jcp.dimK;

    auto test_cond_dimK_reg_block = [](jit_conv_winograd_conf_t jcp,
            int dimK_reg_block, int current_best) {
        // Avoid code size too big
        return (dimK_reg_block <= 64) && (dimK_reg_block < current_best);
    };
    auto test_cond_dimK_nb_block = [](jit_conv_winograd_conf_t jcp,
            int dimK_nb_block, int current_best) {

        int M_thread_size = jcp.oc * jcp.alpha * jcp.alpha
            * jcp.dimK / dimK_nb_block * sizeof(float);;
        int V_thread_size = jcp.ic * jcp.alpha * jcp.alpha
            * jcp.dimK / dimK_nb_block * sizeof(float);;
        int U_size = jcp.oc * jcp.ic * jcp.alpha * jcp.alpha * sizeof(float);

        return ((V_thread_size + M_thread_size + U_size
                    >= C2_1 * L2_cache_size)
                && (V_thread_size + M_thread_size + U_size
                    <= C2_2 * L2_cache_size)
                && dimK_nb_block >= T1 * omp_get_max_threads()
                && dimK_nb_block <= T2 * omp_get_max_threads());
    };
    auto test_cond_dimN_block = [](jit_conv_winograd_conf_t jcp,
            int dimN_block, int current_best) {
        float ratio = .3; //jcp.dimN * 1. / (jcp.dimN + jcp.dimM);
        int V_block_size = jcp.dimK_reg_block * jcp.dimK_block
            * jcp.dimN_reg_block * dimN_block * sizeof(float);
        return ((V_block_size >= C1_1 * ratio * L1_cache_size)
                && (V_block_size <= C1_2 * ratio * L1_cache_size));
    };
    auto test_cond_dimM_block = [](jit_conv_winograd_conf_t jcp,
            int dimM_block, int current_best) {
        float ratio = 0.8; //jcp.dimM * 1. / (jcp.dimN + jcp.dimM);
        int M_block_size = jcp.dimM_simd_block * dimM_block
            * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
        int U_block_size = jcp.dimM_simd_block * dimM_block
            * jcp.dimN_reg_block * jcp.dimN_block * sizeof(float);
        return ((M_block_size + U_block_size >= C1_1 * ratio * L1_cache_size)
                && (M_block_size + U_block_size <= C1_2 * ratio * L1_cache_size));
    };

    bool meet_K_blocking = false;
    for (int i = 0; i < 10; i++) {
        dimK_reg_block = get_max_divisor_satisfying_cond(
                jcp, jcp.dimK, current_dimK_reg_block, test_cond_dimK_reg_block);
        if (dimK_reg_block == current_dimK_reg_block) // fail
            break;
        dimK_nb_block = get_max_divisor_satisfying_cond(jcp,
                jcp.dimK / dimK_reg_block, current_dimK_nb_block,
                test_cond_dimK_nb_block);
        if (dimK_nb_block == current_dimK_nb_block) { // fail
            current_dimK_nb_block = dimK_nb_block;
            current_dimK_reg_block = dimK_reg_block;
        } else {
            meet_K_blocking = true;
            break;
        }
    }

    if (!meet_K_blocking)
        return status::unimplemented;

    // TODO: 4fma -wxy
    jcp.dimK_reg_block = dimK_reg_block;
    jcp.dimK_nb_block = dimK_nb_block;
    jcp.dimK_block = jcp.dimK / jcp.dimK_4fma / jcp.dimK_reg_block / jcp.dimK_nb_block;

    // TODO: tailing handling for ZMM parallelism -wxy
    jcp.dimN_reg_block = jcp.ic_simd_block; // ZMM parallel
    jcp.dimN_block = get_max_divisor_satisfying_cond(jcp,
            jcp.dimN / jcp.dimN_reg_block,
            1, test_cond_dimN_block);
    //jcp.dimN_block = 2;
    jcp.dimN_nb_block = jcp.dimN / jcp.dimN_block / jcp.dimN_reg_block;

    jcp.dimM_simd_block = jcp.oc_simd_block;
    jcp.dimM_block = get_max_divisor_satisfying_cond(jcp,
            jcp.dimM / jcp.dimM_simd_block,
            1, test_cond_dimM_block);
    //jcp.dimM_block = 2;
    jcp.dimM_nb_block = jcp.dimM / jcp.dimM_block / jcp.dimM_simd_block;


    int M_thread_size = jcp.oc * jcp.alpha * jcp.alpha
        * jcp.dimK_block * jcp.dimK_reg_block * sizeof(float);;
    int V_thread_size = jcp.ic * jcp.alpha * jcp.alpha
        * jcp.dimK_block * jcp.dimK_reg_block * sizeof(float);;
    int U_size = jcp.oc * jcp.ic * jcp.alpha * jcp.alpha * sizeof(float);
    printf("M_thread_size=%d, V_thread_size=%d, U_size=%d\n",
            M_thread_size, V_thread_size, U_size);

    int M_block_size = jcp.dimM_simd_block * jcp.dimM_block
        * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
    int V_block_size = jcp.dimN_reg_block * jcp.dimN_block
        * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
    int U_block_size = jcp.dimN_reg_block * jcp.dimN_block
        * jcp.dimM_simd_block * jcp.dimM_block * sizeof(float);
    printf("M_gemm_size=%d, V_gemm_size=%d, U_gemm_size=%d\n",
            M_block_size, V_block_size, U_block_size);

    jcp.sched_policy = WSCHED_WEI_SDGt_W;
    printf("set sched policy WEI_SDGt_W\n");

    return status::success;
}

status_t set_wsched_WEI_SDGit_W(jit_conv_winograd_conf_t &jcp)
{
    return status::unimplemented;

    //jcp.sched_policy = WSCHED_WEI_D_SGit_W;
    //printf("set sched policy WEI_D_SGit_W\n");
    //return status::success;
}

status_t set_wsched_WEI_SDGot_W(jit_conv_winograd_conf_t &jcp)
{
    /*
       Parameter selection: 
       Same as SGD_W but with additional thread-blocking via M

       1. V:thread-size + M:thread-size + U:size: [C1, C2] * L2_cache_size
       2. work-amount is within (T1 ~ T2) * OMP-MAX-THREADS (thread balance)
       3. V:N-block-size + M:M-block-size + U:M-block-size < C * L1_cache_size

       */ 

    const float C1_1 = 0.01, C1_2 = 1.1; // L1
    const float C2_1 = .2, C2_2 = .9; // L2, overflow to L3
    const int T1 = 3, T2 = 256;

    auto test_cond_dimK_reg_block = [](jit_conv_winograd_conf_t jcp,
            int dimK_reg_block, int current_best) {
        // Avoid code size too big
        return (dimK_reg_block <= 64) && (dimK_reg_block < current_best);
    };
    auto test_cond_dimK_nb_block = [](jit_conv_winograd_conf_t jcp,
            int dimK_nb_block, int current_best) {

        int M_thread_size = jcp.dimM / jcp.dimM_nb_block
            * jcp.alpha * jcp.alpha
            * jcp.dimK / dimK_nb_block * sizeof(float);;
        int V_thread_size = jcp.ic * jcp.alpha * jcp.alpha
            * jcp.dimK / dimK_nb_block * sizeof(float);;
        int U_size = jcp.dimM / jcp.dimM_nb_block
            * jcp.ic * jcp.alpha * jcp.alpha * sizeof(float);

        return ((V_thread_size + M_thread_size + U_size
                    >= C2_1 * L2_cache_size)
                && (V_thread_size + M_thread_size + U_size
                    <= C2_2 * L2_cache_size)
                && dimK_nb_block >= T1 * omp_get_max_threads()
                && dimK_nb_block <= T2 * omp_get_max_threads());
    };
    auto test_cond_dimN_block = [](jit_conv_winograd_conf_t jcp,
            int dimN_block, int current_best) {
        float ratio = .9; //jcp.dimN * 1. / (jcp.dimN + jcp.dimM);
        int V_block_size = jcp.dimK_reg_block * jcp.dimK_block
            * jcp.dimN_reg_block * dimN_block * sizeof(float);
        int M_block_size = jcp.dimM_simd_block * jcp.dimM_block
            * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
        int U_block_size = jcp.dimM_simd_block * jcp.dimM_block
            * jcp.dimN_reg_block * jcp.dimN_block * sizeof(float);

        return ((V_block_size + M_block_size + U_block_size
                    >= C1_1 * ratio * L1_cache_size)
                && (V_block_size + M_block_size + U_block_size
                    <= C1_2 * ratio * L1_cache_size));
    };
    auto test_cond_dimM_nb_block = [](jit_conv_winograd_conf_t jcp,
            int dimM_nb_block, int current_best) {
        return dimM_nb_block > current_best;
    };

    bool meet_condition = false;

    int current_dimM_nb_block = 1;
    for (int j = 0; j < 2; ++j) {
        jcp.dimM_nb_block = get_min_divisor_satisfying_cond(
                jcp, jcp.dimM / jcp.oc_simd_block, current_dimM_nb_block,
                test_cond_dimM_nb_block);
        if (jcp.dimM_nb_block == current_dimM_nb_block) { // fail
            break;
        }

        current_dimM_nb_block = jcp.dimM_nb_block;

        int dimK_reg_block = jcp.dimK, current_dimK_reg_block = jcp.dimK;
        int dimK_nb_block = jcp.dimK, current_dimK_nb_block = jcp.dimK;
        for (int i = 0; i < 10; i++) {
            dimK_reg_block = get_max_divisor_satisfying_cond(
                    jcp, jcp.dimK, current_dimK_reg_block, test_cond_dimK_reg_block);
            if (dimK_reg_block == current_dimK_reg_block) // fail
                break;
            current_dimK_reg_block = dimK_reg_block;
            dimK_nb_block = get_max_divisor_satisfying_cond(jcp,
                    jcp.dimK / dimK_reg_block, current_dimK_nb_block,
                    test_cond_dimK_nb_block);
            if (dimK_nb_block == current_dimK_nb_block) // fail
                continue;
            current_dimK_nb_block = dimK_nb_block;

            jcp.dimK_reg_block = dimK_reg_block;
            jcp.dimK_nb_block = dimK_nb_block;
            jcp.dimK_block = jcp.dimK / jcp.dimK_4fma / jcp.dimK_reg_block / jcp.dimK_nb_block;

            jcp.dimM_simd_block = jcp.oc_simd_block;
            //jcp.dimM_block = 2;
            jcp.dimM_block = jcp.dimM / jcp.dimM_nb_block / jcp.dimM_simd_block;

            // TODO: tailing handling for ZMM parallelism -wxy
            jcp.dimN_reg_block = jcp.ic_simd_block; // ZMM parallel
            jcp.dimN_block = get_max_divisor_satisfying_cond(jcp,
                    jcp.dimN / jcp.dimN_reg_block,
                    jcp.dimN, test_cond_dimN_block);
            if (jcp.dimN_block != jcp.dimN) {
                meet_condition = true;
                jcp.dimN_nb_block = jcp.dimN / jcp.dimN_block / jcp.dimN_reg_block;
                goto meet;
            }
        }
    }

meet:
    if (!meet_condition)
        return status::unimplemented;

    int M_thread_size = jcp.dimM / jcp.dimM_nb_block
        * jcp.alpha * jcp.alpha
        * jcp.dimK / jcp.dimK_nb_block * sizeof(float);;
    int V_thread_size = jcp.ic * jcp.alpha * jcp.alpha
        * jcp.dimK / jcp.dimK_nb_block * sizeof(float);;
    int U_size = jcp.dimM / jcp.dimM_nb_block
        * jcp.ic * jcp.alpha * jcp.alpha * sizeof(float);

    printf("M_thread_size=%d, V_thread_size=%d, U_size=%d\n",
            M_thread_size, V_thread_size, U_size);fflush(NULL);

    int M_block_size = jcp.dimM_simd_block * jcp.dimM_block
        * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
    int V_block_size = jcp.dimN_reg_block * jcp.dimN_block
        * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
    int U_block_size = jcp.dimN_reg_block * jcp.dimN_block
        * jcp.dimM_simd_block * jcp.dimM_block * sizeof(float);
    printf("M_gemm_size=%d, V_gemm_size=%d, U_gemm_size=%d\n",
            M_block_size, V_block_size, U_block_size);

    jcp.sched_policy = WSCHED_WEI_SDGot_W;
    printf("set sched policy WEI_SDGot_W\n");

    return status::success;
}

status_t set_wsched_WEI_S_D_Giot_W(jit_conv_winograd_conf_t &jcp)
{
    jcp.dimN_reg_block = jcp.ic_simd_block;
    jcp.dimN_nb_block = 1;
    jcp.dimN_block = jcp.dimN / jcp.dimN_reg_block / jcp.dimN_nb_block;

    jcp.dimM_simd_block = jcp.oc_simd_block;
    jcp.dimM_nb_block = 1;
    jcp.dimM_block = (jcp.dimM / jcp.dimM_simd_block) / jcp.dimM_nb_block;

    const float C2_1 = .1; const float C2_2 = .5;
    auto test_cond_dimK_reg_block = [](jit_conv_winograd_conf_t jcp,
            int dimK_reg_block, int current_best) {

        int M_gemm_size = jcp.dimM_simd_block * jcp.dimM_block
            * dimK_reg_block * jcp.dimK_block * sizeof(float);
        int V_gemm_size = jcp.dimN_reg_block * jcp.dimN_block
            * dimK_reg_block * jcp.dimK_block * sizeof(float);
        int U_gemm_size = jcp.dimN_reg_block * jcp.dimN_block
            * jcp.dimM_simd_block * jcp.dimM_block * sizeof(float);

        return (dimK_reg_block < current_best
                && (V_gemm_size + M_gemm_size + U_gemm_size
                    >= C2_1 * L2_cache_size)
                && (V_gemm_size + M_gemm_size + U_gemm_size
                    <= C2_2 * L2_cache_size));
    };

    int current_dimK_reg_block = 28;
    jcp.dimK_reg_block = get_max_divisor_satisfying_cond(
            jcp, jcp.dimK, current_dimK_reg_block, test_cond_dimK_reg_block);
    if (jcp.dimK_reg_block == current_dimK_reg_block)
        return status::unimplemented;        

    jcp.dimK_block = 1;
    jcp.dimK_nb_block = jcp.dimK / jcp.dimK_4fma / jcp.dimK_reg_block / jcp.dimK_block;

    int M_gemm_size = jcp.dimM_simd_block * jcp.dimM_block
        * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
    int V_gemm_size = jcp.dimN_reg_block * jcp.dimN_block
        * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
    int U_gemm_size = jcp.dimN_reg_block * jcp.dimN_block
        * jcp.dimM_simd_block * jcp.dimM_block * sizeof(float);

    printf("M_gemm_size=%d, V_gemm_size=%d, U_gemm_size=%d\n",
            M_gemm_size, V_gemm_size, U_gemm_size);

    jcp.sched_policy = WSCHED_WEI_S_D_Giot_W;
    printf("set sched policy WEI_S_D_Giot_W\n");

    return status::success;
 
}


status_t set_wsched_WEI_S_D_G_W(jit_conv_winograd_conf_t &jcp)
{
    /*************** Choose dimN_reg_block (ic_simd_block)
     * *******************************/
    /*Hardcoded to 16 because N = ic for bwd weights and
     innermost dimension for ic is assumed 16 in src transforms. This
     choice covers load latencies while maintaining simplicity of kernel
     for POR topologies. FIXME in future??: Will not work for future topologies
     when ic%16 != 0*/
    jcp.dimN_reg_block = jcp.ic_simd_block;

    /****************************** Choose dimK_block
     * **************************/
    // No freedom for choosing dimM_simd_block because ic_simd_block
    // is determined by input data format
    jcp.dimM_simd_block = jcp.oc_simd_block;

    // TODO: tuning params -wxy
#if 0
    auto test_cond1bis_dimK_block = [](
            jit_conv_winograd_conf_t jcp, int dimK_block, int current_best) {
        return check_cond1bis_wu(1, jcp.dimM_simd_block, dimK_block, 1,
                       jcp.dimK_4fma, jcp.dimN_reg_block, 0.4f)
                && (dimK_block > current_best);
    };

    auto test_cond1_dimK_block = [](
            jit_conv_winograd_conf_t jcp, int dimK_block, int current_best) {
        return check_cond1_wu(1, jcp.dimM_simd_block, dimK_block, 1,
                       jcp.dimK_4fma, jcp.dimN_reg_block, 0.4f)
                && (dimK_block > current_best);
    };

    auto test_cond2bis_dimK_block = [](
            jit_conv_winograd_conf_t jcp, int dimK_block, int current_best) {
        return check_cond2bis_wu(1, jcp.dimM_simd_block, dimK_block, 1,
                       jcp.dimK_4fma, 1, jcp.dimN_reg_block, 0.5f)
                && (dimK_block > current_best);
    };

    auto test_cond2_dimK_block = [](
            jit_conv_winograd_conf_t jcp, int dimK_block, int current_best) {
        return check_cond2_wu(1, jcp.dimM_simd_block, dimK_block, 1,
                       jcp.dimK_4fma, 1, jcp.dimN_reg_block, 0.1f)
                && (dimK_block > current_best);
    };

    jcp.dimK_block = get_divisor_satisfying_cond(
            jcp, jcp.dimK / jcp.dimK_4fma, 1, test_cond2bis_dimK_block);
    if (jcp.dimK_block < jcp.dimK / jcp.dimK_4fma)
        jcp.dimK_block = get_divisor_satisfying_cond(
                jcp, jcp.dimK / jcp.dimK_4fma, 1, test_cond2_dimK_block);

    jcp.dimK_reg_block = get_divisor_satisfying_cond(
            jcp, jcp.dimK_block, 1, test_cond1bis_dimK_block);
    if (jcp.dimK_reg_block < jcp.dimK_block) {
        jcp.dimK_reg_block = get_divisor_satisfying_cond(
                jcp, jcp.dimK_block, 1, test_cond1_dimK_block);
    }
#endif
    jcp.dimK_reg_block = 16;
    //jcp.dimK_block /= jcp.dimK_reg_block;
    jcp.dimK_block = 1;
    jcp.dimK_nb_block
            = jcp.dimK / jcp.dimK_4fma / jcp.dimK_reg_block / jcp.dimK_block;

    /***************************** Chose dimN block
     * ****************************/
    auto test_cond2_dimN_block = [](
            jit_conv_winograd_conf_t jcp, int dimN_block, int current_best) {
        return check_cond2_wu(1, jcp.dimM_simd_block, jcp.dimK_block,
                       jcp.dimK_reg_block, jcp.dimK_4fma, dimN_block,
                       jcp.dimN_reg_block, 0.5f)
                && (dimN_block > current_best);
    };

#if 0
    jcp.dimN_block = get_divisor_satisfying_cond(
            jcp, jcp.dimN / jcp.dimN_reg_block, 1, test_cond2_dimN_block);
#endif
    jcp.dimN_block = 16;
    jcp.dimN_nb_block = jcp.dimN / jcp.dimN_reg_block / jcp.dimN_block;

    /********************************* Choose dimM block
     * ************************/

    auto test_cond1_dimM_block = [](
            jit_conv_winograd_conf_t jcp, int dimM_block, int current_best) {
        return check_cond1_wu(dimM_block, jcp.dimM_simd_block, 1,
                       jcp.dimK_reg_block, jcp.dimK_4fma, jcp.dimN_reg_block,
                       1.0f)
                && (dimM_block > current_best)
                && (jcp.dimM / jcp.dimM_simd_block / dimM_block) >= 2;
    };

    jcp.dimM_block = get_divisor_satisfying_cond(
            jcp, jcp.dimM / jcp.dimM_simd_block, 1, test_cond1_dimM_block);
    //jcp.dimM_block = 16;
    jcp.dimM_nb_block = (jcp.dimM / jcp.dimM_simd_block) / jcp.dimM_block;

    int M_gemm_size = jcp.dimM_simd_block * jcp.dimM_block
        * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
    int V_gemm_size = jcp.dimN_reg_block * jcp.dimN_block
        * jcp.dimK_reg_block * jcp.dimK_block * sizeof(float);
    int U_gemm_size = jcp.dimN_reg_block * jcp.dimN_block
        * jcp.dimM_simd_block * jcp.dimM_block * sizeof(float);
    printf("M_gemm_size=%d, V_gemm_size=%d, U_gemm_size=%d\n",
            M_gemm_size, V_gemm_size, U_gemm_size);

    jcp.sched_policy = WSCHED_WEI_S_D_G_W;
    printf("set sched policy WEI_S_D_G_W\n");

    return status::success;

}


status_t jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &diff_dst_d,
        const memory_desc_wrapper &diff_weights_d)
{
    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = 16;

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
    jcp.r_pad = nstl::max(
            0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);
    jcp.b_pad = nstl::max(
            0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih - jcp.t_pad);
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.with_bias = (cd.diff_bias_desc.format != memory_format::undef);

    jcp.ver = mayiuse(avx512_mic_4ops) ? ver_4fma : ver_fma;

    // Winograd specific initialization
    const int tile_size = jcp.alpha - 2;
    jcp.itiles = (jcp.ow + tile_size - 1) / tile_size;
    jcp.jtiles = (jcp.oh + tile_size - 1) / tile_size;
    jcp.ntiles = jcp.mb * jcp.itiles * jcp.jtiles;

    // Winograd kernel works only for 3x3 convolution with stride 1
    if (jcp.ngroups != 1)
        return status::unimplemented;
    if ((jcp.kh != 3) || (jcp.kw != 3))
        return status::unimplemented;
    if ((jcp.stride_h != 1) || (jcp.stride_w != 1))
        return status::unimplemented;
    if ((jcp.ic % simd_w) != 0 || (jcp.oc % simd_w) != 0)
        return status::unimplemented;
    if (src_d.format() != nChw16c)
        return status::unimplemented;
    if (diff_weights_d.format() != (with_groups ? gOIhw16i16o : OIhw16i16o))
        return status::unimplemented;
    if (diff_dst_d.format() != nChw16c)
        return status::unimplemented;

    /*************************** New Kernel Parameters
     * *****************************/
    jcp.ic_simd_block = simd_w;
    jcp.oc_simd_block = simd_w;
    jcp.dimK_4fma = 1;
    jcp.tile_4fma_padding = 0;

#define MAX_4FMA_UR 8
    if (jcp.ver == ver_4fma) {
        auto test_cond_4fma = [](
                jit_conv_winograd_conf_t jcp, int dimK_4fma, int current_best) {
            return (dimK_4fma % 4 == 0) && (dimK_4fma <= MAX_4FMA_UR)
                    && (dimK_4fma > current_best);
        };
        jcp.dimK_4fma = get_divisor_satisfying_cond(
                jcp, jcp.itiles * jcp.jtiles, 4, test_cond_4fma);
        if (jcp.dimK_4fma == 1)
            jcp.dimK_4fma = 4;
        if ((jcp.itiles * jcp.jtiles) % jcp.dimK_4fma != 0)
            jcp.tile_4fma_padding = jcp.dimK_4fma
                    - ((jcp.itiles * jcp.jtiles) % jcp.dimK_4fma);
    }

    jcp.tile_4fma = jcp.dimK_4fma;
    /*NOTE: When (itiles * jtiles) % dimK_4fma != 0, transpose in diff_src
     * transform
     * will not work correctly, this is solved by applying padding.*/
    jcp.dimK = jcp.mb * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding);
    jcp.dimN = jcp.ic;
    jcp.dimM = jcp.oc;

    jcp.double_buffering = true;
    if (jcp.double_buffering)
        jcp.zmm_start = jcp.ver == ver_4fma ? 8 : 2;
    else
        jcp.zmm_start = jcp.ver == ver_4fma ? 4 : 1;
    jcp.nb_reg = 32 - jcp.zmm_start;

    status_t res;
    jcp.sched_policy = WSCHED_INVALID;
    if ((res = set_wsched_WEI_SDGt_W(jcp))   == status::success ||
        (res = set_wsched_WEI_SDGot_W(jcp))  == status::success ||
        (res = set_wsched_WEI_SDGit_W(jcp))  == status::success ||
        (res = set_wsched_WEI_S_D_Giot_W(jcp)) == status::success ||
        (res = set_wsched_WEI_S_D_G_W(jcp)) == status::success)
        ;

    jcp.tile_block_ur = jcp.dimK_reg_block;
    jcp.nb_tile_block_ur = jcp.dimK_block;
    jcp.tile_block = jcp.dimK_nb_block;

    jcp.ic_block = jcp.dimN_block;
    jcp.nb_ic = jcp.dimN_nb_block;

    jcp.oc_block = jcp.dimM_block;
    jcp.nb_oc = jcp.dimM_nb_block;

    printf("dimN_nb_block=%d, dimN_block=%d, dimN_reg_block=%d\n",
            jcp.dimN_nb_block, jcp.dimN_block, jcp.dimN_reg_block);
    printf("dimK_nb_block=%d, dimK_block=%d, dimK_reg_block=%d\n",
            jcp.dimK_nb_block, jcp.dimK_block, jcp.dimK_reg_block);
    printf("dimM_nb_block=%d, dimM_block=%d, dimM_simd_block=%d\n",
            jcp.dimM_nb_block, jcp.dimM_block, jcp.dimM_simd_block);


    return res;

}
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
