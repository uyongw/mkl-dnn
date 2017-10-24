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

#ifndef CPU_JIT_AVX512_COMMON_CONVOLUTION_WINOGRAD_HPP
#define CPU_JIT_AVX512_COMMON_CONVOLUTION_WINOGRAD_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "scratchpad.hpp"

#include "jit_avx512_common_conv_winograd_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {
inline void allocate_winograd_scratchpad(const jit_conv_winograd_conf_t &jcp,
        size_t &up_offset, size_t &vp_offset,
        size_t &mp_offset, size_t &bp_offset,
        scratchpad_t *&winograd_scratchpad)
{
    size_t up_size = 0, vp_size = 0, mp_size = 0, bp_size = 0;
    if (jcp.sched_policy == WSCHED_DATA_W_SGDt) {
        up_size = jcp.alpha * jcp.alpha * jcp.ic * jcp.oc * sizeof(float);
        vp_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.ic * jcp.tile_4fma * sizeof(float);
        mp_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.oc * jcp.tile_4fma * sizeof(float);
    } else if (jcp.sched_policy == WSCHED_DATA_W_S_GDot) {
        up_size = jcp.alpha * jcp.alpha * jcp.ic * jcp.oc * sizeof(float);
        vp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.ic * jcp.mb * sizeof(float);
        mp_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.oc_simd_block * jcp.oc_block * jcp.tile_4fma * sizeof(float);
    } else if (jcp.sched_policy == WSCHED_WEI_SDGt_W) {
        up_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * jcp.ic * jcp.oc * sizeof(float);
        vp_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.ic * jcp.tile_4fma * sizeof(float);
        mp_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.oc * jcp.tile_4fma * sizeof(float);
        bp_size = omp_get_max_threads() * jcp.oc * sizeof(float);
    } else if (jcp.sched_policy == WSCHED_WEI_SDGtWo) {
        up_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * jcp.oc_block * jcp.oc_simd_block * jcp.ic * sizeof(float);
        vp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.ic * jcp.mb * sizeof(float);
        mp_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.oc_simd_block * jcp.oc_block * jcp.tile_4fma * sizeof(float);
        bp_size = omp_get_max_threads() * jcp.oc * sizeof(float);
    } else if (jcp.sched_policy == WSCHED_WEI_S_D_Giot_W) {
        up_size = omp_get_max_threads() * jcp.alpha * jcp.alpha
            * jcp.ic * jcp.oc * sizeof(float);
        vp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.ic * jcp.mb * sizeof(float);
        mp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.oc * jcp.mb * sizeof(float);
        bp_size = omp_get_max_threads() * jcp.oc * sizeof(float);
    } else {
        assert(jcp.sched_policy == WSCHED_DATA_W_S_G_D
                || jcp.sched_policy == WSCHED_WEI_S_D_G_W);
        up_size = jcp.alpha * jcp.alpha * jcp.ic * jcp.oc * sizeof(float);
        vp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.ic * jcp.mb * sizeof(float);
        mp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.oc * jcp.mb * sizeof(float);
        if (jcp.sched_policy == WSCHED_WEI_S_D_G_W)
            bp_size = omp_get_max_threads() * jcp.oc * sizeof(float);
    }

    /* Allocating memory buffers on a page boundary reduces TLB/page misses */
    const size_t page_size = 2097152;
    up_offset = 0;
    vp_offset = utils::rnd_up(up_size, page_size);
    mp_offset = vp_offset + utils::rnd_up(vp_size, page_size);
    bp_offset = mp_offset + utils::rnd_up(mp_size, page_size);

    winograd_scratchpad = create_scratchpad(bp_offset + bp_size);
}
}

template <bool with_relu>
struct _jit_avx512_common_convolution_winograd_fwd_t : public cpu_primitive_t {
    struct pd_t : public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {
        }

        DECLARE_COMMON_PD_T(
                _jit_avx512_common_convolution_winograd_fwd_t<with_relu>);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->cdesc_().prop_kind, forward_training,
                               forward_inference)
                    && this->cdesc_().alg_kind == alg_kind::convolution_winograd
                    && utils::everyone_is(data_type::f32,
                               this->cdesc_().src_desc.data_type,
                               this->cdesc_().weights_desc.data_type,
                               this->cdesc_().dst_desc.data_type)
                    && utils::implication(this->with_bias(), data_type::f32
                                       == this->cdesc_().bias_desc.data_type);
            if (!ok)
                return status::unimplemented;

            return jit_avx512_common_conv_winograd_fwd_kernel_f32::init_conf(
                    jcp_, this->cdesc_(), *this->src_pd_.desc(),
                    *this->weights_pd_.desc(), *this->dst_pd_.desc(), with_relu,
                    this->negative_slope());
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override
        {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nChw16c));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nChw16c));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(
                        this->with_groups() ? gOIhw16i16o : OIhw16i16o));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _jit_avx512_common_convolution_winograd_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs)
        , conf_(*pd)
        , kernel_(nullptr)
        , up_offset_(0)
        , vp_offset_(0)
        , mp_offset_(0)
        , scratchpad_buffer_(nullptr)
    {
        const auto &jcp = conf_.jcp_;
        kernel_ = new jit_avx512_common_conv_winograd_fwd_kernel_f32(conf_.jcp_);
        size_t dummy_bp_offset;
        allocate_winograd_scratchpad(
                jcp, up_offset_, vp_offset_, mp_offset_,
                dummy_bp_offset, scratchpad_buffer_);
    }

    ~_jit_avx512_common_convolution_winograd_fwd_t()
    {
        delete kernel_;
        delete scratchpad_buffer_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();

    void _execute_forward_W_S_G_D();
    void _execute_forward_W_S_GDot();
    void _execute_forward_W_SGDt();

    pd_t conf_;
    jit_avx512_common_conv_winograd_fwd_kernel_f32 *kernel_;

    // Buffer required to store transforms in the frequency domain
    scratchpad_t *scratchpad_buffer_;
    size_t up_offset_;
    size_t vp_offset_;
    size_t mp_offset_;
};

using jit_avx512_common_convolution_winograd_fwd_t
        = _jit_avx512_common_convolution_winograd_fwd_t<false>;
using jit_avx512_common_convolution_winograd_relu_t
        = _jit_avx512_common_convolution_winograd_fwd_t<true>;

struct jit_avx512_common_convolution_winograd_bwd_data_t
        : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {
        }

        DECLARE_COMMON_PD_T(jit_avx512_common_convolution_winograd_bwd_data_t);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->desc()->prop_kind, backward_data)
                    && this->desc()->alg_kind == alg_kind::convolution_winograd
                    && utils::everyone_is(data_type::f32,
                               this->desc()->diff_src_desc.data_type,
                               this->desc()->weights_desc.data_type,
                               this->desc()->diff_dst_desc.data_type);
            if (!ok)
                return status::unimplemented;

            return jit_avx512_common_conv_winograd_bwd_data_kernel_f32::
                    init_conf(jcp_, *this->desc(), *this->diff_src_pd_.desc(),
                            *this->weights_pd_.desc(),
                            *this->diff_dst_pd_.desc());
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override
        {
            using namespace memory_format;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(nChw16c));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nChw16c));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(
                        this->with_groups() ? gOIhw16i16o : OIhw16i16o));
            return status::success;
        }
    };

    jit_avx512_common_convolution_winograd_bwd_data_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs)
        , conf_(*pd)
        , kernel_(nullptr)
        , up_offset_(0)
        , vp_offset_(0)
        , mp_offset_(0)
        , scratchpad_buffer_(nullptr)
    {
        const auto &jcp = conf_.jcp_;
        kernel_ = new jit_avx512_common_conv_winograd_bwd_data_kernel_f32(jcp);
        size_t dummy_bp_offset;
        allocate_winograd_scratchpad(
                jcp, up_offset_, vp_offset_, mp_offset_,
                dummy_bp_offset, scratchpad_buffer_);
    }

    ~jit_avx512_common_convolution_winograd_bwd_data_t()
    {
        delete kernel_;
        delete scratchpad_buffer_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward_data: execute_backward_data(); break;
        default: assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data();
    void _execute_backward_data_W_S_G_D();
    void _execute_backward_data_W_SGDt();

    pd_t conf_;
    jit_avx512_common_conv_winograd_bwd_data_kernel_f32 *kernel_;

    // Buffer required to store transforms in the frequency domain
    scratchpad_t *scratchpad_buffer_;
    size_t up_offset_;
    size_t vp_offset_;
    size_t mp_offset_;
};

struct jit_avx512_common_convolution_winograd_bwd_weights_t
        : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {
        }

        DECLARE_COMMON_PD_T(jit_avx512_common_convolution_winograd_bwd_weights_t);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->desc()->prop_kind, backward_weights)
                    && this->desc()->alg_kind == alg_kind::convolution_winograd
                    && utils::everyone_is(data_type::f32,
                               this->desc()->src_desc.data_type,
                               this->desc()->diff_dst_desc.data_type,
                               this->desc()->diff_weights_desc.data_type);
            if (!ok)
                return status::unimplemented;

            return jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::
                    init_conf(jcp_, *this->desc(), *this->src_pd_.desc(),
                            *this->diff_dst_pd_.desc(),
                            *this->diff_weights_pd_.desc());
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override
        {
            using namespace memory_format;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nChw16c));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nChw16c));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(
                        this->with_groups() ? gOIhw16i16o : OIhw16i16o));
            if (diff_bias_pd_.desc()->format == any)
                CHECK(diff_bias_pd_.set_format(x));
            return status::success;
        }
    };

    jit_avx512_common_convolution_winograd_bwd_weights_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs)
        , conf_(*pd)
        , kernel_(nullptr)
        , up_offset_(0)
        , vp_offset_(0)
        , mp_offset_(0)
        , bp_offset_(0)
        , scratchpad_buffer_(nullptr)
    {
        auto jcp = conf_.jcp_;
        kernel_ = new jit_avx512_common_conv_winograd_bwd_weights_kernel_f32(
                conf_.jcp_);
        allocate_winograd_scratchpad(
                jcp, up_offset_, vp_offset_, mp_offset_, bp_offset_, scratchpad_buffer_);
    }

    ~jit_avx512_common_convolution_winograd_bwd_weights_t()
    {
        delete kernel_;
        delete scratchpad_buffer_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward_weights: execute_backward_weights(); break;
        default: assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights();
    void _execute_backward_weights_S_D_G_W();
    void _execute_backward_weights_S_D_Giot_W();
    void _execute_backward_weights_SDGtWo();
    void _execute_backward_weights_SDGt_W();

    pd_t conf_;
    jit_avx512_common_conv_winograd_bwd_weights_kernel_f32 *kernel_;

    // Buffer required to store transforms in the frequency domain
    scratchpad_t *scratchpad_buffer_;
    size_t up_offset_;
    size_t vp_offset_;
    size_t mp_offset_;
    size_t bp_offset_;
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
