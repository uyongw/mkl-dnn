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

#include <assert.h>

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "math_utils.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"
#include "cpu_barrier.hpp"

#include "jit_uni_batch_normalization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {

using namespace Xbyak;
namespace barrier = simple_barrier;

#define  BN_BLOCKING_ALWAYS    (0)
#define  BN_BLOCKING_ENABLE    (1)

typedef float data_t;
static long g_llc_size = 0;

template <cpu_isa_t isa>
struct jit_bnorm_t: public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        size_t N_ithr, N_nthr;
        size_t coff_max, soff_max;
        size_t mb_stride_Bc, spat_size;
        data_t chan_size, eps, one;
        const data_t *scale_shift;
        const data_t *mean, *var;
        const data_t *diff_scale_shift;
        const data_t *src, *dst;
        const data_t *diff_src, *diff_dst;
        const data_t *rbuf1, *rbuf2;
        barrier::ctx_t *barrier;
    };
    /*used when fuse with ReLU layer.*/
    unsigned with_relu;
    double negative_slope;

    /* cpu specific part */
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;
    const AddressFrame &vmmword = (isa == sse42) ? xword :
                                  (isa == avx2) ? yword : zword;

    void uni_vpxor(const Xmm& x1, const Xmm& x2, const Operand& op) {
        if (isa == sse42) pxor(x2, op);
        else if (isa == avx2) vpxor(x1, x2, op);
        else vpxord(x1, x2, op);
    }

    const int vlen = isa == sse42 ? 32 : cpu_isa_traits<isa>::vlen;

    const batch_normalization_pd_t *bdesc_;

    void (*ker)(const call_params_t *);
    void operator()(const call_params_t *p) { (*ker)(p); }

    Reg64 reg_param = abi_param1;

    Reg64 reg_scale_shift = rbx;
    Reg64 reg_rbuf1 = abi_not_param1;
    Reg64 reg_rbuf2 = rdx;

    Reg64 reg_mean = rbp;
    Reg64 reg_var = reg_param;
    Reg64 reg_diff_scale_shift = rax;

    Reg64 reg_coff = r8;
    Reg64 reg_coff_max = r9;
    Reg64 reg_soff = r10;
    Reg64 reg_soff_max = r11;
    Reg64 reg_ctr = r12;
    Reg64 reg_roff = r13;

    Reg64 reg_mb_stride_Bc = r14;

    Reg64 reg_src = r15;
    Reg64 reg_diff_src = reg_rbuf1;
    Reg64 reg_dst = rsi;
    Reg64 reg_diff_dst = reg_dst;

    Reg64 simd_iter = reg_dst;
    Reg64 dst_simd_iter = rcx;
    Reg64 reg_tmp_off = reg_roff;

    // Reuse loop counters
    Reg64 reg_bar = reg_coff;
    Reg64 reg_nnthr = reg_soff; // must be usable w/ loops over coff
    Reg64 reg_tmp = reg_ctr;

    size_t unroll_blocks = isa == avx512_common ? 4 : 1;
    size_t unroll_regs = isa == avx512_common ? 4 : 1;
    Vmm vbuf = Vmm(isa == avx512_common ? 20 : 5);
    Vmm vdiff_beta = Vmm(isa == avx512_common ? 21 : 6);
    Vmm vdiff_gamma = Vmm(isa == avx512_common ? 22 : 7);
    Vmm vsqrtvar = Vmm(isa == avx512_common ? 23 : 8);
    Vmm vone = Vmm(isa == avx512_common ? 24 : 9);
    Vmm vmean = Vmm(isa == avx512_common ? 25 : 10);
    Vmm vvar = Vmm(isa == avx512_common ? 26 : 11);
    Vmm vgamma = Vmm(isa == avx512_common ? 27 : 12);
    Vmm vbeta = Vmm(isa == avx512_common ? 28 : 13);
    Vmm veps = Vmm(isa == avx512_common ? 29 : 14);
    Vmm vchan_size = Vmm(isa == avx512_common ? 31 : 15);

    size_t t0_pf_offt;
    size_t t1_pf_offt;
    size_t spat_size;
    size_t chan_data_offt;

    enum {
        stack_off_N_nthr = 0,
        stack_off_N_ithr = 8,
        stack_off_src = 16,
        stack_off_dst = 24,
        stack_off_diff_src = 32,
        stack_off_diff_dst = 40,
        stack_off_barrier = 48,
    };

    void compute_static_strides() {
        spat_size = bdesc_->W() * bdesc_->H();
        chan_data_offt = bdesc_->C() * sizeof(data_t);

        if (isa == avx512_mic) {
            t0_pf_offt = 4096;
            t1_pf_offt = 0;
        } else {
            t0_pf_offt = 0;
            t1_pf_offt = 0;
        }
    }

    void load_common_params() {
#       define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_rbuf1, ptr[reg_param + PARAM_OFF(rbuf1)]);
        if (bdesc_->is_bwd())
            mov(reg_rbuf2, ptr[reg_param + PARAM_OFF(rbuf2)]);
        mov(reg_coff_max, ptr[reg_param + PARAM_OFF(coff_max)]);
        mov(reg_soff_max, ptr[reg_param + PARAM_OFF(soff_max)]);
        mov(reg_mb_stride_Bc, ptr[reg_param + PARAM_OFF(mb_stride_Bc)]);
        shl(reg_coff_max, 2);
        shl(reg_soff_max, 2);
        shl(reg_mb_stride_Bc, 2);

        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_scale_shift, ptr[reg_param + PARAM_OFF(scale_shift)]);

        uni_vbroadcastss(vchan_size, vmmword[reg_param + PARAM_OFF(chan_size)]);
        uni_vbroadcastss(vone, vmmword[reg_param + PARAM_OFF(one)]);
        uni_vbroadcastss(veps, vmmword[reg_param + PARAM_OFF(eps)]);

        mov(reg_tmp, ptr[reg_param + PARAM_OFF(N_nthr)]);
        mov(ptr[rsp + stack_off_N_nthr], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(N_ithr)]);
        mov(ptr[rsp + stack_off_N_ithr], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(src)]);
        mov(ptr[rsp + stack_off_src], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(dst)]);
        mov(ptr[rsp + stack_off_dst], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_src)]);
        mov(ptr[rsp + stack_off_diff_src], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_dst)]);
        mov(ptr[rsp + stack_off_diff_dst], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(barrier)]);
        mov(ptr[rsp + stack_off_barrier], reg_tmp);

        if (bdesc_->is_fwd()) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(var)]);
            mov(reg_var, reg_tmp);
        } else {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_scale_shift)]);
            mov(reg_diff_scale_shift, reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(var)]);
            mov(reg_var, reg_tmp);
        }
#       undef PARAM_OFF
    }

    void barrier() {
        mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
        mov(reg_bar, ptr[rsp + stack_off_barrier]);
        simple_barrier::generate(*this, reg_bar, reg_nnthr);
    }

    Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + reg_coff + offt + 0 * chan_data_offt];
    }

    Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + reg_coff + offt + 0 * chan_data_offt];
    }

    Address diff_gamma_ptr(size_t offt = 0) {
        return vmmword[reg_diff_scale_shift + reg_coff + offt
            + 0 * chan_data_offt];
    }

    Address diff_beta_ptr(size_t offt = 0) {
        return vmmword[reg_diff_scale_shift + reg_coff + offt
            + 1 * chan_data_offt];
    }

    Address gamma_ptr(size_t offt = 0) {
        return vmmword[reg_scale_shift + reg_coff + offt + 0 * chan_data_offt];
    }

    Address beta_ptr(size_t offt = 0) {
        return vmmword[reg_scale_shift + reg_coff + offt + 1 * chan_data_offt];
    }

    template <typename init_t, typename body_t, typename fini_t>
    void spat_loop(size_t len, size_t blocks, size_t regs,
            init_t init, body_t body, fini_t fini) {
        size_t factor = regs * blocks;
        size_t loop_unroll = len / factor * factor;
        size_t loop_tail = len - loop_unroll;
        size_t num_active_regs = (len < regs) ? len : regs;

        for (size_t i = 0; i < num_active_regs; i++)
            init(i);

        if (loop_unroll) {
            mov(reg_ctr, loop_unroll);
            Label label;
            L(label); {
                for (size_t i = 0; i < factor; i++) {
                    size_t base_reg = i % regs;
                    body(base_reg, i);
                }
                add(reg_soff, factor * vlen);
                sub(reg_ctr, factor);
                jnz(label);
            }
        }

        for (size_t i = 0; i < loop_tail; i++) {
            size_t base_reg = i % regs;
            body(base_reg, i);
        }
        if (loop_tail)
            add(reg_soff, loop_tail * vlen);

        for (size_t i = 0; i < num_active_regs; i++)
            fini(i);
    }

    void mean_channels() {
        Label ch_label;
        L(ch_label); {
            uni_vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
            spat_loop(spat_size, unroll_blocks,
                unroll_regs,
                    [=](size_t base_reg) {
                        Vmm v = Vmm(base_reg * 2);
                        if (base_reg)
                            uni_vpxor(v, v, v);
                    },
                    [=](size_t base_reg, size_t i) {
                        Vmm v0 = Vmm(base_reg * 2 + 0);
                        Vmm v1 = Vmm(base_reg * 2 + 1);
                        size_t offt = i * vlen;
                        uni_vmovups(v1,
                            vmmword[reg_src + reg_soff + offt]);
                        uni_vaddps(v0, v0, v1);
                        mic_prefetcht0(ptr[reg_src + reg_soff + offt
                                + t0_pf_offt]);
                        mic_prefetcht1(ptr[reg_src + reg_soff + offt
                                + t1_pf_offt]);
                    },
                    [=](size_t base_reg) {
                        Vmm b = Vmm(0);
                        Vmm v = Vmm(base_reg * 2);
                        if (base_reg)
                            uni_vaddps(b, b, v);
                    });
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));

            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void var_channels() {
        Label ch_label;
        L(ch_label); {
            uni_vmovups(vmean, mean_ptr());
            uni_vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
            spat_loop(spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                        Vmm v = Vmm(base_reg * 3);
                        if (base_reg > 0)
                            uni_vpxor(v, v, v);
                    },
                    [=](size_t base_reg, size_t i) {
                        Vmm v = Vmm(3 * base_reg);
                        Vmm vtmp0 = Vmm(3 * base_reg + 1);
                        Vmm vtmp1 = Vmm(3 * base_reg + 2);
                        size_t offt = i * vlen;
                        uni_vmovups(vtmp0,
                            vmmword[reg_src + reg_soff + offt]);
                        if (isa == sse42) {
                            movups(vtmp1, vmean);
                            subps(vtmp1, vtmp0);
                        } else {
                            vsubps(vtmp1, vmean, vtmp0);
                        }
                        uni_vfmadd231ps(v, vtmp1, vtmp1);

                        mic_prefetcht0(ptr[reg_src + reg_soff + offt
                                + t0_pf_offt]);
                        mic_prefetcht1(ptr[reg_src + reg_soff + offt
                                + t1_pf_offt]);
                    },
                    [=](size_t base_reg) {
                        Vmm b = Vmm(0);
                        Vmm v = Vmm(base_reg * 3);
                        if (base_reg)
                            uni_vaddps(b, b, v);
                    });
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void compute_mean_variance() {
        uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
        xor_(reg_coff, reg_coff);
        Label zero_rbuf;
        L(zero_rbuf); {
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            add(reg_coff, isa == sse42 ? vlen / 2 : vlen);
            cmp(reg_coff, reg_coff_max);
            jne(zero_rbuf);
        }

        mov(reg_src, ptr[rsp + stack_off_src]);

        xor_(reg_soff, reg_soff);
        Label mean_spatial;
        L(mean_spatial); {
            xor_(reg_coff, reg_coff);

            if (isa == sse42)
                mov(reg_tmp_off, reg_soff);

            mean_channels();

            if (isa == sse42) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);

                mean_channels();

                sub(reg_src, vlen / 2);
            }

            add(reg_soff, reg_mb_stride_Bc);
            cmp(reg_soff, reg_soff_max);
            jne(mean_spatial);
        }

        Label no_mean_reduction;
        barrier(); {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            jne(no_mean_reduction);
            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            Label mean_reduction_channels;
            L(mean_reduction_channels); {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                mov(reg_ctr, reg_nnthr);
                Label mean_reduction_thrs;
                L(mean_reduction_thrs); {
                    uni_vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf1 + reg_roff]);
                    uni_vmovups(vmmword[reg_rbuf1 + reg_roff], Vmm(0));
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(mean_reduction_thrs);
                }
                uni_vdivps(Vmm(1), Vmm(1), vchan_size);
                uni_vmovups(mean_ptr(), Vmm(1));

                add(reg_coff, isa == sse42 ? vlen / 2 : vlen);

                cmp(reg_coff, reg_coff_max);
                jne(mean_reduction_channels);
            }
        }
        L(no_mean_reduction);
        barrier();

        xor_(reg_soff, reg_soff);
        Label var_spatial;
        L(var_spatial); {
            xor_(reg_coff, reg_coff);

            if (isa == sse42)
                mov(reg_tmp_off, reg_soff);

            var_channels();

            if (isa == sse42) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);

                var_channels();

                sub(reg_src, vlen / 2);
            }

            add(reg_soff, reg_mb_stride_Bc);
            cmp(reg_soff, reg_soff_max);
            jne(var_spatial);
        }

        Label no_var_reduction;
        barrier(); {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            jne(no_var_reduction);

            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            Label var_reduction_channels;
            L(var_reduction_channels); {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                mov(reg_ctr, reg_nnthr);
                Label var_reduction_thrs;
                L(var_reduction_thrs); { // TODO: unroll (?)
                    uni_vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf1 + reg_roff]);
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(var_reduction_thrs);
                }
                uni_vdivps(Vmm(1), Vmm(1), vchan_size);
                uni_vmovups(var_ptr(), Vmm(1));
                add(reg_coff, isa == sse42 ? vlen / 2 : vlen);

                cmp(reg_coff, reg_coff_max);
                jne(var_reduction_channels);
            }
        }
        L(no_var_reduction);
        barrier();
    }

    void forward_channels() {
        Label ch_label;
        L(ch_label); {
            uni_vmovups(vmean, mean_ptr());
            uni_vmovups(vsqrtvar, var_ptr());
            uni_vaddps(vsqrtvar, vsqrtvar, veps);
            uni_vsqrtps(vsqrtvar, vsqrtvar);

            if (isa == sse42) {
                movups(vbuf, vone);
                divps(vbuf, vsqrtvar);
                movups(vsqrtvar, vbuf);
            } else {
                vdivps(vsqrtvar, vone, vsqrtvar);
            }

            if (bdesc_->use_scaleshift()) {
                uni_vmovups(vgamma, gamma_ptr());
                uni_vmovups(vbeta, beta_ptr());
            }
            /* prepare the zero and theta register. :  vmm5*/
            Vmm  vmm_slope;
            if(with_relu) {
                if(avx512_common) {
                    assert(unroll_blocks<=4); 
                    assert(unroll_regs<=4);
                }
                if(!avx512_common) {
                    assert(unroll_blocks<=1); 
                    assert(unroll_regs<=1);
                }
                if(avx512_common){
                    vmm_slope=Vmm(5);
                    /*here need use reg_ctr(reg_tmp) to swap, make sure its safe to use reg_tmp*/
                    mov(reg_ctr, float2int(negative_slope));
                    movq(Xmm(5), reg_ctr);
                    uni_vbroadcastss(vmm_slope, Xmm(5));
                } else {
                     vmm_slope=Vmm(2);
                     /*here need use reg_ctr(reg_tmp) to swap, make sure its safe to use reg_tmp*/
                     mov(reg_ctr, float2int(negative_slope));
                     movq(Xmm(2), reg_ctr);
                     uni_vbroadcastss(vmm_slope, Xmm(2));
                }

            }
            spat_loop(spat_size, unroll_blocks, unroll_regs/*4:  loop 4 time on vmm0~vmm3.*/,
                    [](size_t base_reg) {UNUSED(base_reg);},
                    [=](size_t base_reg, size_t i) {
                        Vmm v = Vmm(base_reg);
                        size_t offt = i * vlen;
                        uni_vmovups(v,
                            vmmword[reg_src + reg_soff + offt]);
                        mic_prefetcht0(ptr[reg_src + reg_soff + offt
                                + t0_pf_offt]);
                        mic_prefetcht1(ptr[reg_src + reg_soff + offt
                                + t1_pf_offt]);
                        uni_vsubps(v, v, vmean);
                        uni_vmulps(v, v, vsqrtvar);
                        if (bdesc_->use_scaleshift()) {
                            uni_vfmadd213ps(v, vgamma, vbeta);
                        }
                        /*******************************************
                          *here do Relu logic on forward path:
                         avx512:
                            1. register needed. 1const reg(slope) + (2vmm additional registers for each unloop)*(unroll_regs)
                                1+2*4=9; 
                              a. BN current, 20~31 but 30 alreay fix used in BN, 
                              b. BN loop unroll alreay used: 1(one reg each time * 4(unroll_regs)= 4regs 
                              c. So ReLU logic can use from 4~19.(16)
                              d. reg mapping:
                                const:                     
                                vmm_slope: for mul,  vmm(5)

                                for each unloop: 
                                vmm_tmp:  for mul, and store the mul result:vmm 6 8 10 12((base_reg+3)*2)
                                mask: for cmp, store the cmp result:        vmm 7 9 11 13((base_reg+3)*2+1)

                                additional: 
                                tmp/mask: vmm14 is used to swap. as vmm(0) is needed in sse42 path.
                            2. how many loop can unroll. 4
                          avx2/sse4_2
                            1. register needed. 1const reg(slope) + (2vmm additional registers for each unloop)*(unroll_regs)
                                1+2*1=3; 
                              a. BN current, 5~15 fix used in BN, only 0~4 case use
                              b. BN loop unroll alreay used: 1(one reg each time * 1(unroll_regs)= 1regs 
                              c. So ReLU logic can use from 1~4.(4)
                              d. reg mapping:
                                const:                     
                                vmm_slope: for mul,  vmm(2)

                                for each unloop: 
                                vmm_tmp:  for mul, and store the mul result:vmm 3((base_reg+3))
                                mask: for cmp, store the cmp result:        vmm 4((base_reg+3)+1)

                                additional: 
                                tmp/mask: vmm1 is used to swap. as vmm(0) is needed in sse42 path.
                            2. how many loop can unroll. 1                          
                            3. input: v   out:xmm_tmp
                          ********************************************/
                        if(with_relu) {
                            unsigned char _cmp_gt_os = isa == avx512_common ? 14 : 6;
                            Vmm  vmm_tmp, vmm_mask; 
                            if(avx512_common) {
                              vmm_tmp=Vmm((base_reg+3)*2);
                              vmm_mask=Vmm((base_reg+3)*2 + 1);
                            } else {
                                vmm_tmp=Vmm((base_reg+3));
                                vmm_mask=Vmm((base_reg+3)+ 1);
                            }
                            if (isa == sse42) {
                                Vmm src=v; 
                                if(base_reg==0) { /*resv xmm(0), as mask need to be xmm0*/
                                    src=Vmm(1);
                                    movups(src, v);
                                }
                                Vmm mask = Vmm(0);
                                
                                uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
                                movups(mask, src);
                                cmpps(mask, vmm_tmp, _cmp_gt_os);
                                movups(vmm_tmp, src);
                                mulps(vmm_tmp, vmm_slope);
                                blendvps(vmm_tmp, src); 

                            } else {
                            
                                uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
                                if (isa == avx2) {
                                    vcmpgtps(vmm_mask, v, vmm_tmp);
                                    vmulps(vmm_tmp, v, vmm_slope); 
                                    vblendvps(vmm_tmp, vmm_tmp, v, vmm_mask);
                                } else {
                                    Opmask k_mask = Opmask(1);
                                    vcmpps(k_mask, v, vmm_tmp, _cmp_gt_os);
                                    vmulps(vmm_tmp, v, vmm_slope); 
                                    vblendmps(vmm_tmp | k_mask, vmm_tmp, v);
                                }
                            }
                            /*Relu end.*/
                            /*write out*/
                            uni_vmovntps(vmmword[reg_dst + reg_soff + offt], vmm_tmp); 
                        } else  {
                            uni_vmovntps(vmmword[reg_dst + reg_soff + offt],
                                v);
                        }
                    },
                    [](size_t base_reg) {UNUSED(base_reg);});

            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void forward() {
        mov(reg_src, ptr[rsp + stack_off_src]);
        mov(reg_dst, ptr[rsp + stack_off_dst]);

        xor_(reg_soff, reg_soff);
        Label dst_spatial;
        L(dst_spatial); {
            xor_(reg_coff, reg_coff);
            if (isa == sse42)
                mov(reg_tmp_off, reg_soff);

            forward_channels();

            if (isa == sse42) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                add(reg_dst, vlen / 2);
                mov(reg_coff, vlen / 2);

                forward_channels();

                sub(reg_src, vlen / 2);
                sub(reg_dst, vlen / 2);
            }

            add(reg_soff, reg_mb_stride_Bc);
            cmp(reg_soff, reg_soff_max);
            jnz(dst_spatial);
        }
    }

    void backward() {
        uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
        xor_(reg_coff, reg_coff);
        Label zero_rbuf, sh_spatial, sh_channels;

        L(zero_rbuf); {
            vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            vmovups(vmmword[reg_rbuf2 + reg_coff], Vmm(0));
            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jne(zero_rbuf);
        }

        mov(reg_src, ptr[rsp + stack_off_src]);
        mov(reg_diff_dst, ptr[rsp + stack_off_diff_dst]);
        xor_(reg_soff, reg_soff);
        L(sh_spatial); {
            xor_(reg_coff, reg_coff);
            L(sh_channels); {
                vmovups(vmean, mean_ptr());
                vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
                vmovups(Vmm(1), vmmword[reg_rbuf2 + reg_coff]);
                spat_loop(spat_size, 1, 1,
                        [=](size_t base_reg) {
                            if (base_reg > 0) {
                                for (int i = 0; i < 2; i++) {
                                    Vmm v(base_reg * 5 + i);
                                    uni_vpxor(v, v, v);
                                }
                            }
                        },
                        [=](size_t base_reg, size_t i) {
                            Vmm o0 = Vmm(base_reg * 5 + 0);
                            Vmm o1 = Vmm(base_reg * 5 + 1);
                            Vmm t1 = Vmm(base_reg * 5 + 2);
                            Vmm t2 = Vmm(base_reg * 5 + 3);
                            Vmm t3 = Vmm(base_reg * 5 + 4);
                            size_t offt = i * vlen;
                            vmovups(t1, vmmword[reg_src + reg_soff + offt]);
                            vmovups(t2, vmmword[reg_diff_dst + reg_soff
                                    + offt]);
                            vsubps(t3, vmean, t1);
                            vfnmadd231ps(o0, t3, t2);
                            vaddps(o1, t2);
                            mic_prefetcht0(ptr[reg_diff_dst + reg_soff + offt
                                    + t0_pf_offt]);
                            mic_prefetcht0(ptr[reg_src + reg_soff + offt
                                    + t0_pf_offt]);
                            mic_prefetcht1(ptr[reg_diff_dst + reg_soff + offt
                                    + t1_pf_offt]);
                            mic_prefetcht1(ptr[reg_src + reg_soff + offt
                                    + t1_pf_offt]);
                        },
                        [=](size_t base_reg) {
                            Vmm b0 = Vmm(0);
                            Vmm b1 = Vmm(1);
                            if (base_reg) {
                                vaddps(b0, b0, Vmm(base_reg * 5 + 0));
                                vaddps(b1, b1, Vmm(base_reg * 5 + 1));
                            }
                        });
                vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
                vmovups(vmmword[reg_rbuf2 + reg_coff], Vmm(1));
                add(reg_coff, vlen);
                cmp(reg_coff, reg_coff_max);
                jne(sh_channels);
            }
            add(reg_soff, reg_mb_stride_Bc);
            cmp(reg_soff, reg_soff_max);
            jne(sh_spatial);
        }

        Label no_sh_reduction;
        barrier(); {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            Label sh_reduction_channels;
            jne(no_sh_reduction);

            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            L(sh_reduction_channels); {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                vmovups(vsqrtvar, var_ptr());
                vaddps(vsqrtvar, vsqrtvar, veps);
                vsqrtps(vsqrtvar, vsqrtvar);
                vdivps(vsqrtvar, vone, vsqrtvar);
                mov(reg_ctr, reg_nnthr);
                Label sh_reduction_thrs;
                L(sh_reduction_thrs); { // TODO: unroll (?)
                    vaddps(Vmm(0), Vmm(0), vmmword[reg_rbuf1 + reg_roff]);
                    vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf2 + reg_roff]);
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(sh_reduction_thrs);
                }
                vmulps(Vmm(0), Vmm(0), vsqrtvar);
                vmovups(diff_gamma_ptr(), Vmm(0));
                vmovups(diff_beta_ptr(), Vmm(1));
                add(reg_coff, vlen);
                cmp(reg_coff, reg_coff_max);
                jne(sh_reduction_channels);
            }
        }
        L(no_sh_reduction);
        barrier();

        mov(reg_diff_src, ptr[rsp + stack_off_diff_src]);
        xor_(reg_soff, reg_soff);
        Label diff_spatial;
        L(diff_spatial); {
            xor_(reg_coff, reg_coff);
            Label diff_channels;
            L(diff_channels); {
                vmovups(vmean, mean_ptr());
                vmovups(vsqrtvar, var_ptr());
                vaddps(vsqrtvar, vsqrtvar, veps);
                vsqrtps(vsqrtvar, vsqrtvar);
                vdivps(vsqrtvar, vone, vsqrtvar);
                if (bdesc_->use_scaleshift()) {
                    vmovups(vgamma, gamma_ptr());
                }
                vmovups(vdiff_gamma, diff_gamma_ptr());
                vmovups(vdiff_beta, diff_beta_ptr());
                vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);
                vdivps(vdiff_beta, vdiff_beta, vchan_size);
                vdivps(vdiff_gamma, vdiff_gamma, vchan_size);

                spat_loop(spat_size, unroll_blocks, unroll_regs,
                        [=](size_t base_reg) {UNUSED(base_reg);},
                        [=](size_t base_reg, size_t i) {
                            Vmm v(base_reg * 2 + 0);
                            Vmm t(base_reg * 2 + 1);
                            size_t offt = i * vlen;
                            vmovups(v, vmmword[reg_diff_dst + reg_soff
                                    + offt]);
                            if (!bdesc_->omit_stats()) {
                                vsubps(v, v, vdiff_beta);
                                vmovups(t, vmmword[reg_src + reg_soff + offt]);
                                vsubps(t, vmean, t);
                                vmulps(t, t, vdiff_gamma);
                                vaddps(v, v, t);
                            }
                            vmulps(v, v, vsqrtvar);
                            if (bdesc_->use_scaleshift()) {
                               vmulps(v, v, vgamma);
                            }
                            vmovntps(vmmword[reg_diff_src + reg_soff + offt],
                                    v);
                            mic_prefetcht0(ptr[reg_diff_dst + reg_soff + offt
                                    + t0_pf_offt]);
                            mic_prefetcht0(ptr[reg_src + reg_soff + offt
                                    + t0_pf_offt]);
                            mic_prefetcht1(ptr[reg_diff_dst + reg_soff
                                    + offt + t1_pf_offt]);
                            mic_prefetcht1(ptr[reg_src + reg_soff + offt
                                    + t1_pf_offt]);
                        },
                        [=](size_t base_reg) {UNUSED(base_reg);});

                add(reg_coff, vlen);
                cmp(reg_coff, reg_coff_max);
                jne(diff_channels);
            }
            add(reg_soff, reg_mb_stride_Bc);
            cmp(reg_soff, reg_soff_max);
            jne(diff_spatial);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *bdesc): bdesc_(bdesc) {
        with_relu=bdesc_->desc()->with_relu;
        negative_slope=bdesc_->desc()->negative_slope;
        //fprintf(stderr,  "<jit_bnorm_t>:with_relu:%d negative_slope:%f\n",with_relu,negative_slope);
        static_assert(isa == sse42 || isa == avx2 || isa == avx512_common
                || isa == avx512_mic, "unsupported isa");

        preamble();
        compute_static_strides();
        sub(rsp, 56);
        load_common_params();
        if (bdesc_->is_fwd()) {
            if (!bdesc_->stats_is_src()) {
                compute_mean_variance();
            }
            forward();
        } else {
            backward();
        }
        add(rsp, 56);
        postamble();

        ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                    this->getCode()));
    }
};

template <cpu_isa_t isa>
struct uni_bnorm_driver_t: public c_compatible {
    uni_bnorm_driver_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc), ker_(bdesc_), syncable_(true), buf_(nullptr)
        , barriers_(nullptr)
    {
        use_tmp_stats_ = !bdesc_->stats_is_src()
            && bdesc_->desc()->prop_kind == prop_kind::forward_inference;
        use_tmp_diff_scale_shift_ = false
            || (bdesc_->is_bwd() && !bdesc_->use_scaleshift())
            || bdesc_->desc()->prop_kind == prop_kind::backward_data;
        int num_sbufs = 2 * use_tmp_stats_;
        int num_pbufs = 2 * use_tmp_diff_scale_shift_;
        int num_rbufs = bdesc_->is_fwd() ? 1 : 2;

        int buf_size =
            (num_sbufs + num_pbufs + num_rbufs * bdesc_->MB()) * bdesc_->C();
        buf_ = (data_t *)malloc(buf_size * sizeof(data_t), 64);

        sbuf_ = buf_;
        pbuf_ = sbuf_ + num_sbufs * bdesc_->C();
        rbuf_ = pbuf_ + num_pbufs * bdesc_->C();

        int num_barriers = bdesc_->C() / simd_w;
        if (syncable_) {
            barriers_ = (barrier::ctx_t *)malloc(
                    num_barriers * sizeof(barrier::ctx_t), 64);
            for (int i = 0; i < num_barriers; ++i)
                barrier::ctx_init(&barriers_[i]);
        }
        if (BN_BLOCKING_ENABLE && !g_llc_size) {
            g_llc_size = get_cache_size(3, false) * get_num_processors();
            if (g_llc_size > 40*1024*1024)
                g_llc_size -= 16*1024*1024;
#if defined(VERBOSE_INFO)
            fprintf(stderr, "[exec]llc_size: %ld\n", g_llc_size);
#endif
        }
        with_relu=bdesc_->desc()->with_relu; 
        negative_slope=bdesc_->desc()->negative_slope; 
    }
    ~uni_bnorm_driver_t() { free(buf_); free(barriers_); }

    void exec(int ithr, int nthr, const data_t *src, data_t *diff_src,
            data_t *dst, const data_t *diff_dst, const data_t *scale_shift,
            data_t *diff_scale_shift, const data_t *mean, const data_t *var) {
        size_t N = bdesc_->MB();
        size_t C = bdesc_->C();
        size_t H = bdesc_->H();
        size_t W = bdesc_->W();
        size_t img_size = C * H * W;

#if defined(VERBOSE_INFO)
        if(ithr==0)
            fprintf(stderr, "[exec]bn: NCHW(%ld,%ld,%ld,%ld)\n", N, C, H, W);
#endif
        if(canBlockEnabled()) {
#if defined(VERBOSE_INFO)
        if(ithr==0)
            fprintf(stderr, "[exec]bn exec block path enabled\n");
#endif
            return exec_block(ithr, nthr, src, diff_src, dst,diff_dst, scale_shift, diff_scale_shift, mean, var);
        }

        typename jit_bnorm_t<isa>::call_params_t p;

        p.eps = bdesc_->desc()->batch_norm_epsilon;
        p.one = 1.;
        p.spat_size = H*W;
        p.chan_size = 1. * N * p.spat_size;

        size_t C_blks = C / simd_w;

        int C_ithr{0}, C_nthr{0}, N_ithr{0}, N_nthr{0};
        size_t C_blk_s{0}, C_blk_e{0}, N_s{0}, N_e{0};
        thread_balance(ithr, nthr, C_blks, C_ithr, C_nthr, C_blk_s, C_blk_e,
                N_ithr, N_nthr, N_s, N_e);

        p.N_ithr = N_ithr;
        p.N_nthr = N_nthr;

        size_t C_blks_thr = C_blk_e - C_blk_s;
        size_t N_thr = N_e - N_s;

        size_t coff_base = C_blk_s * simd_w;
        size_t soff_base = C_blk_s * p.spat_size * simd_w + N_s * img_size;

        p.coff_max = C_blks_thr * simd_w;
        p.mean = (use_tmp_stats_ ? sbuf_ : mean) + coff_base;
        p.var = (use_tmp_stats_ ? sbuf_ + C : var) + coff_base;
        p.scale_shift = scale_shift + coff_base;
        p.diff_scale_shift = (use_tmp_diff_scale_shift_
                ? pbuf_ : diff_scale_shift) + coff_base;

        p.soff_max = N_thr * img_size;
        p.src = src + soff_base;
        p.dst = dst + soff_base;
        p.diff_src = diff_src + soff_base;
        p.diff_dst = diff_dst + soff_base;

        p.mb_stride_Bc = img_size - p.coff_max * p.spat_size;

        p.rbuf1 = rbuf_ + (C_blk_s * N_nthr + p.N_ithr * C_blks_thr) * simd_w;
        p.rbuf2 = p.rbuf1 + C * N_nthr;

        p.barrier = barriers_ + C_ithr;

        if (p.soff_max != 0 && p.coff_max != 0) ker_(&p);
    }

    inline bool canBlockEnabled() {
        size_t N = bdesc_->MB();
        size_t C = bdesc_->C();
        size_t H = bdesc_->H();
        size_t W = bdesc_->W();
        if(BN_BLOCKING_ALWAYS || (BN_BLOCKING_ENABLE && (bdesc_->is_fwd() ? N*C*H*W*sizeof(data_t) : (N*C*H*W*sizeof(data_t)) << 1) >= g_llc_size)) {
            return true;
        }
        return false;
    }

    // div whole task accord to C(16c) into blocks, futher div each block accord to N to available core#.
    void exec_block(int ithr, int nthr, const data_t *src, data_t *diff_src,
            data_t *dst, const data_t *diff_dst, const data_t *scale_shift,
            data_t *diff_scale_shift, const data_t *mean, const data_t *var) {
        size_t N = bdesc_->MB();
        size_t C = bdesc_->C();
        size_t H = bdesc_->H();
        size_t W = bdesc_->W();
        size_t img_size = C * H * W;

        typename jit_bnorm_t<isa>::call_params_t p;

        p.eps = bdesc_->desc()->batch_norm_epsilon;
        p.one = 1.;
        p.spat_size = H*W;
        p.chan_size = 1. * N * p.spat_size; // all batch. all value of same channel in mini-batch, non 16 C mode.

        size_t C_blks = C / simd_w;

        int C_ithr{0}, C_nthr{0}, N_ithr{0}, N_nthr{0};
        size_t C_blk_s{0}, C_blk_e{0}, N_s{0}, N_e{0};
        int c_bksPerIter, iters;

        inCache_computing_balance(nthr, C_blks, c_bksPerIter, iters);
#if defined(VERBOSE_INFO)
        if(ithr==0) fprintf(stderr, "[exec] c_bksPerIter:%d iters:%d \n", c_bksPerIter,iters);
#endif

        //balance in each iter.
        thread_balance_for_block(ithr, nthr, c_bksPerIter, C_ithr, C_nthr, C_blk_s, C_blk_e,
            N_ithr, N_nthr, N_s, N_e);
        p.N_ithr = N_ithr;
        p.N_nthr = N_nthr;

        int it, last_iter_blks = C_blks - (iters - 1) * c_bksPerIter;
        //replace C_blk_s with global C_blk_s
        size_t global_C_blk_s;
        size_t global_barriers_per_iter = C_nthr;
        for(it=0; it<iters; it++) { // each thread need work for each blocks, in each block it need to sync.
            if (it == iters-1 && iters > 1) {
                C_blk_s = C_blk_e = N_s = N_e = 0;
                thread_balance_for_block(ithr, nthr, last_iter_blks, C_ithr, C_nthr, C_blk_s, C_blk_e,
                    N_ithr, N_nthr, N_s, N_e);
                p.N_ithr = N_ithr;
                p.N_nthr = N_nthr;
            }

#if defined(VERBOSE_INFO)
            if(ithr==0) {
                fprintf(stderr, "[exec] nthr: %d	N_nthr:%d  C_nthr:%d \n",nthr,N_nthr,C_nthr);
                fprintf(stderr, "[exec] C_ithr:%d C_nthr:%d C_blk_s:%ld C_blk_e:%ld N_ithr:%d N_nthr:%d N_s:%ld N_e:%ld\n",
                    C_ithr, C_nthr, C_blk_s, C_blk_e,	 N_ithr, N_nthr, N_s, N_e);
            }
#endif

            global_C_blk_s = (C_blk_s == -1) ? -1 : it * c_bksPerIter + C_blk_s;

            //follow two arg are encoded into coff_max and soff_max, then to control when to finish the computing.
            size_t C_blks_thr = C_blk_e - C_blk_s;// C num of this thread.
            size_t N_thr = N_e - N_s; // N num of this thread.
            size_t coff_base = global_C_blk_s * simd_w; // channel idx begin non 16C mode.
            size_t soff_base = global_C_blk_s * p.spat_size * simd_w + N_s * img_size; // spat offset. non 16c mode.

            p.coff_max = C_blks_thr * simd_w; // channel idx, non 16c mode, from  0.
            p.mean = (use_tmp_stats_ ? sbuf_ : mean) + coff_base;
            p.var = (use_tmp_stats_ ? sbuf_ + C : var) + coff_base;
            p.scale_shift = scale_shift + coff_base;
            p.diff_scale_shift = (use_tmp_diff_scale_shift_
                ? pbuf_ : diff_scale_shift) + coff_base;

            p.soff_max = N_thr * img_size; // spat offset., NCHW, non  16c mode.
            p.src = src + soff_base;
            p.dst = dst + soff_base;
            p.diff_src = diff_src + soff_base;
            p.diff_dst = diff_dst + soff_base;

            p.mb_stride_Bc = img_size - p.coff_max * p.spat_size; // how many buffer there to stride to next n

            p.rbuf1 = rbuf_ + (global_C_blk_s * N_nthr + p.N_ithr * C_blks_thr) * simd_w;
            p.rbuf2 = p.rbuf1 + C * N_nthr; //end buffer.

            p.barrier = barriers_ + C_ithr + it  *global_barriers_per_iter; // a barrie for each  C thread.  two 16c may share a single barrier. as they are computed at same time.
            if (p.soff_max != 0 && p.coff_max != 0) ker_(&p);
        }
    }

    /* how many iterations are needed, and how many jobs each iteration do. */
    inline void inCache_computing_balance(int nthr, size_t C_blks, int &c_blksPerIter, int &iters) {
        const size_t N = bdesc_->MB();
        const size_t H = bdesc_->H();
        const size_t W = bdesc_->W();
        size_t sizeBasicBlock = bdesc_->is_fwd() ? N * H * W * simd_w * sizeof(data_t) : (N * H * W * simd_w * sizeof(data_t)) << 1;

        c_blksPerIter = g_llc_size / sizeBasicBlock;

        if(c_blksPerIter == 0) c_blksPerIter = 1;
        if(c_blksPerIter > C_blks) c_blksPerIter = C_blks;

        iters = (C_blks + c_blksPerIter - 1) / c_blksPerIter;
    }

private:
    inline void thread_balance(int ithr, int nthr, size_t C_blks, int &C_ithr,
            int &C_nthr, size_t &C_blk_s, size_t &C_blk_e, int &N_ithr,
            int &N_nthr, size_t &N_s, size_t &N_e) const {
        const size_t N = bdesc_->MB();
        if (nthr <= (int)C_blks || !syncable_) {
            C_ithr = ithr; C_nthr = nthr;
            N_ithr = 0; N_nthr = 1;
            N_s = 0; N_e = N;
            C_ithr = ithr; C_nthr = nthr;
            balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
        } else {
            C_nthr = math::gcd(nthr, (int)C_blks);
            N_nthr = nstl::min((int)N, nthr / C_nthr);
            if (ithr < C_nthr * N_nthr) {
                N_ithr = ithr % N_nthr;
                C_ithr = ithr / N_nthr;
                balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
                balance211(N, N_nthr, N_ithr, N_s, N_e);
            } else {
                N_ithr = C_ithr = -ithr;
                N_s = N_e = C_blk_s = C_blk_e = -1;
            }
        }
    }

    inline void thread_balance_for_block(int ithr, int nthr, size_t C_blks, int &C_ithr,
            int &C_nthr, size_t &C_blk_s, size_t &C_blk_e, int &N_ithr,
            int &N_nthr, size_t &N_s, size_t &N_e) const {
        const size_t N = bdesc_->MB();
        if (nthr <= (int)C_blks || !syncable_) {
            C_ithr = ithr; C_nthr = nthr;
            N_ithr = 0; N_nthr = 1;
            N_s = 0; N_e = N;
            C_ithr = ithr; C_nthr = nthr;
            balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
        } else {
            N_nthr = nstl::min((int)N, nthr);
            C_nthr = nstl::min((int)C_blks, nthr / N_nthr);
            if (ithr < C_nthr * N_nthr) {
                N_ithr = ithr % N_nthr;
                C_ithr = ithr / N_nthr;
                balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
                balance211(N, N_nthr, N_ithr, N_s, N_e);
            } else {
                N_ithr = C_ithr = -ithr;
                N_s = N_e = C_blk_s = C_blk_e = -1;
            }
        }
    }

    const int simd_w = isa == sse42 ? 8 :
        cpu_isa_traits<isa>::vlen / sizeof(data_t);

    const batch_normalization_pd_t *bdesc_;
    jit_bnorm_t<isa> ker_;
    bool syncable_;
    bool use_tmp_stats_, use_tmp_diff_scale_shift_;

    data_t *buf_, *sbuf_, *rbuf_, *pbuf_;
    barrier::ctx_t *barriers_;

    public:
    /*used when fused with Relu*/
    unsigned with_relu;
    double negative_slope;
};

}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::jit_uni_batch_normalization_fwd_t(
        const pd_t *pd, const input_vector &inputs,
        const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{ bnorm_driver_ = new uni_bnorm_driver_t<isa>(&conf_); }

template <cpu_isa_t isa>
void jit_uni_batch_normalization_fwd_t<isa>::execute(event_t *e) {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    auto mean = reinterpret_cast<data_t*>(conf_.stats_is_src()
            ? const_cast<char*>(this->input_memory(1))
            : this->memory(1));
    auto var = reinterpret_cast<data_t*>(conf_.stats_is_src()
            ? const_cast<char*>(this->input_memory(2))
            : this->memory(2));

    auto idx_scale_shift = 1 + 2*conf_.stats_is_src();
    auto scale_shift =
        reinterpret_cast<const data_t *>(this->input_memory(idx_scale_shift));

#   pragma omp parallel
    {
        bnorm_driver_->exec(omp_get_thread_num(), omp_get_num_threads(), src,
                nullptr, dst, nullptr, scale_shift, nullptr, mean, var);
    }
    e->set_state(event_t::ready);
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::~jit_uni_batch_normalization_fwd_t() {
    delete bnorm_driver_;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::jit_uni_batch_normalization_bwd_t(
        const pd_t *pd, const input_vector &inputs,
        const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{ bnorm_driver_ = new uni_bnorm_driver_t<isa>(&conf_); }

template <cpu_isa_t isa>
void jit_uni_batch_normalization_bwd_t<isa>::execute(event_t *e) {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto mean = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto var = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(3));
    auto scale_shift = reinterpret_cast<const data_t *>(this->input_memory(4));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));
    auto diff_scale_shift = reinterpret_cast<data_t *>(this->memory(1));

#   pragma omp parallel
    {
        bnorm_driver_->exec(omp_get_thread_num(), omp_get_num_threads(), src,
                diff_src, nullptr, diff_dst, scale_shift, diff_scale_shift,
                mean, var);
    }
    e->set_state(event_t::ready);
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::~jit_uni_batch_normalization_bwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_fwd_t<sse42>;
template struct jit_uni_batch_normalization_fwd_t<avx2>;
template struct jit_uni_batch_normalization_bwd_t<avx2>;
template struct jit_uni_batch_normalization_fwd_t<avx512_common>;
template struct jit_uni_batch_normalization_bwd_t<avx512_common>;

}
}
}
