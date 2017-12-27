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

#define PAGE_SIZE (2 * 1024 * 1024)
// TODO: remove workspace and mmap facility. No use now.
// Alloc zero-filled pages
class Mmap {
public:
	static char *alloc(size_t size)
	{
		size = utils::rnd_up(size, PAGE_SIZE);
#ifdef MAP_ANONYMOUS
		const int mode = MAP_PRIVATE | MAP_ANONYMOUS;
#elif defined(MAP_ANON)
		const int mode = MAP_PRIVATE | MAP_ANON;
#endif
		void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, mode, -1, 0);

		return (p == MAP_FAILED) ? nullptr : (char*)p;
	}

	static void free(char *p, size_t size)
	{
		if (p == 0) return;
		munmap((void*)p, size);
	}
};

namespace winograd {

#define WSP_U_PRIVATE 0x01
#define WSP_V_PRIVATE 0x02
#define WSP_M_PRIVATE 0x04
#define WSP_B_PRIVATE 0x08
struct workspace {
public:
    workspace(size_t up_size, size_t vp_size,
              size_t mp_size, size_t bp_size,
              int max_threads_num, int flags = 0)
        : up_size_(up_size), vp_size_(vp_size),
          mp_size_(mp_size), bp_size_(bp_size),
          nthreads(max_threads_num), flags_(flags)
    {
        const size_t page_size = PAGE_SIZE;
        size_t total_sp_size = 0;
        unsigned long eigen = 0;

        if (flags_ & WSP_U_PRIVATE) {
            up_ = (char*)Mmap::alloc(up_size_);
        } else {
            up_offset_ = 0;
            total_sp_size += utils::rnd_up(up_size, page_size);
        }
        if (flags_ & WSP_V_PRIVATE) {
            vp_ = (char*)Mmap::alloc(vp_size_);
        } else {
            vp_offset_ = total_sp_size;
            total_sp_size += utils::rnd_up(vp_size, page_size);
        }
        if (flags_ & WSP_M_PRIVATE) {
            mp_ = (char*)Mmap::alloc(mp_size_);
        } else {
            mp_offset_ = total_sp_size;
            total_sp_size += utils::rnd_up(mp_size, page_size);
        }
        if (flags_ & WSP_B_PRIVATE) {
            bp_ = (char*)Mmap::alloc(bp_size_);
        } else {
            bp_offset_ = total_sp_size;
            total_sp_size += bp_size;
        }
        if (get_num_processors() > 1) {
            // TODO: Improve eigen, consider sched_policy,
            // ic, oc and ntiles
            eigen = up_size + vp_size + mp_size + bp_size;
        }

        scratchpad_ = create_scratchpad(total_sp_size, eigen);
    }

    ~workspace() {
        if (scratchpad_ != nullptr)
            delete scratchpad_;
        if (up_ != nullptr)
            Mmap::free(up_, up_size_);
        if (vp_ != nullptr)
            Mmap::free(vp_, vp_size_);
        if (mp_ != nullptr)
            Mmap::free(mp_, mp_size_);
        if (bp_ != nullptr)
            Mmap::free(bp_, bp_size_);
    }

    char *up() {
        return up_ == nullptr
            ? scratchpad_->get() + up_offset_ : up_;
    }
    char *vp() {
        return vp_ == nullptr
            ? scratchpad_->get() + vp_offset_ : vp_;
    }
    char *mp() {
        return mp_ == nullptr
            ? scratchpad_->get() + mp_offset_ : mp_;
    }
    char *bp() {
        return bp_ == nullptr
            ? scratchpad_->get() + bp_offset_ : bp_;
    }

    int nthreads = 0;

private:
    size_t up_offset_;
    size_t vp_offset_;
    size_t mp_offset_;
    size_t bp_offset_;

    size_t up_size_ = 0;
    size_t vp_size_ = 0;
    size_t mp_size_ = 0;
    size_t bp_size_ = 0;

    char *up_ = nullptr;
    char *vp_ = nullptr;
    char *mp_ = nullptr;
    char *bp_ = nullptr;

    scratchpad_t *scratchpad_;
    int flags_ = 0;
};

inline void allocate_winograd_workspace(const jit_conv_winograd_conf_t &jcp,
        workspace *&wsp)
{
    size_t up_size = 0, vp_size = 0, mp_size = 0, bp_size = 0;
    int nthreads = omp_get_max_threads();
    int nb_tg = jcp.tg_i * jcp.tg_o * jcp.tg_t;

    switch (jcp.sched_policy) {
    case WSCHED_DATA_W_SGDt:
        up_size = jcp.alpha * jcp.alpha * jcp.ic * jcp.oc * sizeof(float);
        vp_size = nthreads * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.ic * jcp.tile_4fma * sizeof(float);
        mp_size = nthreads * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.oc * jcp.tile_4fma * sizeof(float);
        wsp = new workspace(up_size, vp_size, mp_size, bp_size, nthreads);
        break;
    case WSCHED_DATA_W_S_GDot:
        up_size = jcp.alpha * jcp.alpha * jcp.ic * jcp.oc * sizeof(float);
        vp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.ic * jcp.mb * sizeof(float);
        mp_size = nthreads * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.oc_simd_block * jcp.oc_block * jcp.tile_4fma * sizeof(float);
        wsp = new workspace(up_size, vp_size, mp_size, bp_size, nthreads);
        break;
    case WSCHED_WEI_SDGt_W:
        up_size = nthreads * jcp.alpha * jcp.alpha
            * jcp.ic * jcp.oc * sizeof(float);
        vp_size = nthreads * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.ic * jcp.tile_4fma * sizeof(float);
        mp_size = nthreads * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.oc * jcp.tile_4fma * sizeof(float);
        bp_size = nthreads * jcp.oc * sizeof(float);
        wsp = new workspace(up_size, vp_size, mp_size, bp_size, nthreads);
        break;
    case WSCHED_WEI_SDGtWo:
        up_size = nthreads * jcp.alpha * jcp.alpha
            * jcp.oc_block * jcp.oc_simd_block * jcp.ic * sizeof(float);
        vp_size = nthreads * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.ic * jcp.tile_4fma * sizeof(float);
        mp_size = nthreads * jcp.alpha * jcp.alpha
            * (jcp.nb_tile_block_ur * jcp.tile_block_ur + jcp.tile_4fma_padding)
            * jcp.oc_simd_block * jcp.oc_block * jcp.tile_4fma * sizeof(float);
        bp_size = nthreads * jcp.oc * sizeof(float);
        wsp = new workspace(up_size, vp_size, mp_size, bp_size, nthreads);
        break;
    case WSCHED_WEI_S_D_Giot_W:
        up_size = (nthreads + 1) * jcp.alpha * jcp.alpha
            * jcp.ic * jcp.oc * sizeof(float);
        vp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.ic * jcp.mb * sizeof(float);
        mp_size = jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.oc * jcp.mb * sizeof(float);
        bp_size = nthreads * jcp.oc * sizeof(float);
        wsp = new workspace(up_size, vp_size, mp_size, bp_size, nthreads);
        break;
    case WSCHED_DATA_W_S_G_D:
    case WSCHED_DATA_W_S_G_D_n:
        nthreads /= nb_tg;
        up_size = nb_tg * jcp.alpha * jcp.alpha * jcp.ic * jcp.oc * sizeof(float);
        vp_size = nb_tg * jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.ic * jcp.mb * sizeof(float);
        mp_size = nb_tg * jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.oc * jcp.mb * sizeof(float);
        wsp = new workspace(up_size, vp_size, mp_size, bp_size, nthreads);
        break;
    case WSCHED_WEI_S_D_G_W:
    case WSCHED_WEI_S_D_G_W_n:
        nthreads /= nb_tg;
        up_size = nb_tg * jcp.alpha * jcp.alpha * jcp.ic * jcp.oc * sizeof(float);
        if (jcp.tg_t > 0) {
            up_size += nb_tg * jcp.tg_i * jcp.ic * jcp.tg_o * jcp.oc
                * jcp.kh * jcp.kw * sizeof(float);
        }
        vp_size = nb_tg * jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.ic * jcp.mb * sizeof(float);
        mp_size = nb_tg * jcp.alpha * jcp.alpha
            * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding)
            * jcp.oc * jcp.mb * sizeof(float);
        bp_size = nb_tg * nthreads * jcp.oc * sizeof(float);
        wsp = new workspace(up_size, vp_size, mp_size, bp_size, nthreads);
        break;
    default:
        assert(!"Unknown Winograd schedule policy!");
        break;
    }
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
        , wsp_(nullptr)
    {
        const auto &jcp = conf_.jcp_;
        kernel_ = new jit_avx512_common_conv_winograd_fwd_kernel_f32(conf_.jcp_);
        allocate_winograd_workspace(jcp, wsp_);
    }

    ~_jit_avx512_common_convolution_winograd_fwd_t()
    {
        delete kernel_;
        delete wsp_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        const auto &jcp = conf_.jcp_;

        switch (jcp.alpha) {
        case 3:
            execute_forward<3>();
            break;
        case 4:
            execute_forward<4>();
            break;
        case 5:
            execute_forward<5>();
            break;
        case 6:
            execute_forward<6>();
            break;
        case 7:
            execute_forward<7>();
            break;
        case 8:
            execute_forward<8>();
            break;
        case 9:
            execute_forward<9>();
            break;
        default:
            assert(!"invalid alpha");
            break;
        }
        e->set_state(event_t::ready);
    }

private:
    template<const int alpha> void execute_forward();

    template<const int alpha> void _execute_forward_W_S_G_D();
    template<const int alpha> void _execute_forward_W_S_G_D_n();
    template<const int alpha> void _execute_forward_W_SGDt();

    pd_t conf_;
    winograd::workspace *wsp_;
    jit_avx512_common_conv_winograd_fwd_kernel_f32 *kernel_;
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
        , wsp_(nullptr)
    {
        const auto &jcp = conf_.jcp_;
        kernel_ = new jit_avx512_common_conv_winograd_bwd_data_kernel_f32(jcp);
        allocate_winograd_workspace(jcp, wsp_);
    }

    ~jit_avx512_common_convolution_winograd_bwd_data_t()
    {
        delete kernel_;
        delete wsp_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        const auto &jcp = conf_.jcp_;
        if (conf_.desc()->prop_kind == prop_kind::backward_data) {
            switch (jcp.alpha) {
            case 3:
                execute_backward_data<3>();
                break;
            case 4:
                execute_backward_data<4>();
                break;
            case 5:
                execute_backward_data<5>();
                break;
            case 6:
                execute_backward_data<6>();
                break;
            case 7:
                execute_backward_data<7>();
                break;
            case 8:
                execute_backward_data<8>();
                break;
            case 9:
                execute_backward_data<9>();
                break;
            default:
                assert(!"invalid alpha");
                break;
            }

        } else {
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    template<const int alpha> void execute_backward_data();
    template<const int alpha> void _execute_backward_data_W_S_G_D();
    template<const int alpha> void _execute_backward_data_W_S_G_D_n();
    template<const int alpha> void _execute_backward_data_W_SGDt();

    pd_t conf_;
    winograd::workspace *wsp_;
    jit_avx512_common_conv_winograd_bwd_data_kernel_f32 *kernel_;
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
        , wsp_(nullptr)
    {
        auto jcp = conf_.jcp_;
        kernel_ = new jit_avx512_common_conv_winograd_bwd_weights_kernel_f32(
                conf_.jcp_);
        allocate_winograd_workspace(jcp, wsp_);
    }

    ~jit_avx512_common_convolution_winograd_bwd_weights_t()
    {
        delete kernel_;
        delete wsp_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        const auto &jcp = conf_.jcp_;
        if (conf_.desc()->prop_kind == prop_kind::backward_weights) {
            switch (jcp.alpha) {
            case 3:
                execute_backward_weights<3>();
                break;
            case 4:
                execute_backward_weights<4>();
                break;
            case 5:
                execute_backward_weights<5>();
                break;
            case 6:
                execute_backward_weights<6>();
                break;
            case 7:
                execute_backward_weights<7>();
                break;
            case 8:
                execute_backward_weights<8>();
                break;
            case 9:
                execute_backward_weights<9>();
                break;
            case 10:
                execute_backward_weights<10>();
                break;
            default:
                assert(!"invalid alpha");
                break;
            }

        } else {
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    template<const int alpha> void execute_backward_weights();
    template<const int alpha> void _execute_backward_weights_S_D_G_W();
    template<const int alpha> void _execute_backward_weights_S_D_G_W_n();
    template<const int alpha> void _execute_backward_weights_S_D_Giot_W();
    template<const int alpha> void _execute_backward_weights_SDGtWo();
    template<const int alpha> void _execute_backward_weights_SDGt_W();

    pd_t conf_;
    winograd::workspace *wsp_;
    jit_avx512_common_conv_winograd_bwd_weights_kernel_f32 *kernel_;
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
