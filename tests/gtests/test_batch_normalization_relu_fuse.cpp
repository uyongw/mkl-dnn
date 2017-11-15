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

#include <cmath>

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"
#include <sys/time.h> 

namespace mkldnn {

struct test_bnrm_sizes_t {
    int mb, c, h, w;
};

struct test_bnrm_formats_t {
    mkldnn::memory::format data_format;
    mkldnn::memory::format diff_format;
};

struct test_bnrm_params_t {
    mkldnn::engine::kind engine_kind;
    test_bnrm_formats_t formats;
    test_bnrm_sizes_t sizes;
    double eps;
    unsigned with_relu;
    double negative_slope;
};
void dump2file(char *filename, float *data, unsigned long size){
  
}
template <typename T, typename A> inline T relu_fwd(T s, A alpha) {
    return s > 0 ? s : static_cast<T>(s * alpha);
}
template <typename T, typename A> inline T relu_bwd(T dd, T s, A alpha) {
    return s > 0 ? dd : static_cast<T>(dd * alpha);
}


template <typename data_t>
void check_bnrm_fwd(const test_bnrm_params_t &p,
        const memory &src, const memory &mean, const memory &variance,
        const memory &weights, const memory &dst, unsigned flags, prop_kind pk)
{
    const bool use_weights = flags & use_scale_shift;
    const bool calculate_stats = !(flags & use_global_stats);
    const bool is_training = (pk == prop_kind::forward_training);

    const data_t *src_data = (const data_t *)src.get_data_handle();
    const data_t *weights_data = use_weights ? (const data_t *)weights.get_data_handle() : nullptr;
    const data_t *mean_data = (!calculate_stats || is_training) ?
           (const data_t *)mean.get_data_handle() : nullptr;
    const data_t *variance_data = (!calculate_stats || is_training) ?
           (const data_t *)variance.get_data_handle() : nullptr;
    const data_t *dst_data = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    test_bnrm_sizes_t bp = p.sizes;
    data_t eps = static_cast<data_t>(1.e-4 * bp.mb * bp.h * bp.w);
    //fprintf(stderr, "NCHW: %d %d %d %d eps:%f size: %dk\n",bp.mb, bp.c, bp.h, bp.w,eps,bp.mb*bp.c* bp.h* bp.w*4/1024); 
    fprintf(stderr, "checking ..  fuse:%d\n", p.with_relu);
#pragma omp parallel for
    for (int c = 0; c < bp.c; c++) {
        data_t ref_mean = calculate_stats ? data_t(0) : mean_data[c];
        data_t ref_variance = calculate_stats ? data_t(0) : variance_data[c];
        if (calculate_stats) {
            for (int n = 0; n < bp.mb; n++)
            for (int h = 0; h < bp.h; h++)
                for (int w = 0; w < bp.w; w++) {
                    int sidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                            + h * bp.w + w;
                ref_mean += src_data[map_index(src_d, sidx)];
            }
            ref_mean /= bp.mb * bp.h * bp.w;
            if (is_training) {
                data_t mean_norm_max = std::max(fabs(mean_data[c]), fabs(ref_mean));
                if (mean_norm_max < eps) mean_norm_max = data_t(1);
		//fprintf(stderr, "mean check: c:%d  my:%f ref:%f\n",c,mean_data[c], ref_mean);
                EXPECT_NEAR((mean_data[c] - ref_mean) / mean_norm_max, 0., eps);
            }

            for (int n = 0; n < bp.mb; n++)
            for (int h = 0; h < bp.h; h++)
                for (int w = 0; w < bp.w; w++) {
                    int sidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                            + h * bp.w + w;
                    data_t tmp = src_data[map_index(src_d, sidx)] - ref_mean;
                    ref_variance += tmp * tmp;
                }
            ref_variance /= bp.mb * bp.h * bp.w;
            if (is_training) {
                data_t variance_norm_max = std::max(fabs(variance_data[c]), fabs(ref_variance));
                if (variance_norm_max < eps) variance_norm_max = data_t(1);
		//fprintf(stderr, "variance check: c:%d  my:%f ref:%f\n",c,variance_data[c], ref_variance);
                EXPECT_NEAR((variance_data[c] - ref_variance) / variance_norm_max, 0., eps);
            }
        }
        data_t ref_sqrt_variance = static_cast<data_t>(sqrt(ref_variance + p.eps));
        data_t ref_rsqrt_variance = data_t(1) / (ref_sqrt_variance);

        if (use_weights) {
            memory::desc weights_d = weights.get_primitive_desc().desc();
            for (int n = 0; n < bp.mb; n++)
            for (int h = 0; h < bp.h; h++)
                for (int w = 0; w < bp.w; w++) {
                    int sdidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                            + h * bp.w + w;
                    data_t ref_dst = weights_data[map_index(weights_d, c)]
                            * (src_data[map_index(src_d, sdidx)]
                            - ref_mean) * ref_rsqrt_variance
                            + weights_data[map_index(weights_d, bp.c + c)];
                    //apply relu logic.
                    if(p.with_relu) {
                        ref_dst=relu_fwd(ref_dst, p.negative_slope);
                    }
                    data_t out = dst_data[map_index(dst_d, sdidx)];
                    data_t norm_max = std::max(fabs(out), fabs(ref_dst));
                    if (norm_max < 10e-3) norm_max = data_t(1);
		    //fprintf(stderr, "dst check: c :%d nhw:%d %d %d  my:%f ref:%f\n",c,n,h, w, out, ref_dst);
                    EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
                }
        } else {
            for (int n = 0; n < bp.mb; n++)
            for (int h = 0; h < bp.h; h++)
                for (int w = 0; w < bp.w; w++) {
                    int sdidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                            + h * bp.w + w;
                    data_t ref_dst = (src_data[map_index(src_d, sdidx)]
                            - ref_mean) * ref_rsqrt_variance;
                    data_t out = dst_data[map_index(dst_d, sdidx)];
                   //apply relu logic.
                    if(p.with_relu) {
                        ref_dst=relu_fwd(ref_dst, p.negative_slope);
                    }
                    data_t norm_max = std::max(fabs(out), fabs(ref_dst));
                    if (norm_max < 10e-3) norm_max = data_t(1);
                    EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
                }
        }
    }
}

template <typename data_t>
void check_bnrm_bwd(const test_bnrm_params_t &p,
        const memory &src, const memory &diff_dst, const memory &mean,
        const memory &variance, const memory &weights, const memory &diff_src,
        const memory &diff_weights, unsigned flags, prop_kind pk)
{
    const bool use_weights = flags & use_scale_shift;
    const bool calculate_diff_stats = !(flags & omit_stats);

    const data_t *src_data = (const data_t *)src.get_data_handle();
    const data_t *weights_data = use_weights ? (const data_t *)weights.get_data_handle() : nullptr;
    const data_t *diff_dst_data = (const data_t *)diff_dst.get_data_handle();
    const data_t *mean_data = (const data_t *)mean.get_data_handle();
    const data_t *variance_data = (const data_t *)variance.get_data_handle();
    const data_t *diff_src_data = (data_t *)diff_src.get_data_handle();
    const data_t *diff_weights_data = (pk == prop_kind::backward) ?
            (data_t *)diff_weights.get_data_handle() : nullptr;

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();
    const memory::desc diff_weights_d = diff_weights.get_primitive_desc().desc();

    test_bnrm_sizes_t bp = p.sizes;

    const data_t eps = static_cast<data_t>(1.e-4 * bp.mb * bp.h * bp.w);

#pragma omp parallel for
    for (int c = 0; c < bp.c; c++) {
        data_t ref_diff_gamma = data_t(0);
        data_t ref_diff_beta = data_t(0);

        auto v_mean = mean_data[c];
        auto v_variance = variance_data[c];
        const data_t sqrt_variance = data_t(1.0 / sqrt(v_variance + p.eps));

        auto gamma = use_weights ? weights_data[map_index(weights_d, c)] : 1;

        for (int n = 0; n < bp.mb; n++)
        for (int h = 0; h < bp.h; h++)
        for (int w = 0; w < bp.w; w++) {
            int sidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                    + h * bp.w + w;
            ref_diff_gamma += (src_data[map_index(src_d, sidx)] - v_mean)
                * diff_dst_data[map_index(diff_dst_d, sidx)];
            ref_diff_beta += diff_dst_data[map_index(diff_dst_d, sidx)];
        }
        ref_diff_gamma *= sqrt_variance;

        if (pk == backward) {
            auto diff_gamma = diff_weights_data[map_index(diff_weights_d, c)];
            data_t norm_max = std::max(fabs(diff_gamma), fabs(ref_diff_gamma));
            if (norm_max < 10e-3) norm_max = data_t(1);
            EXPECT_NEAR((diff_gamma - ref_diff_gamma) / norm_max, 0., eps);

            auto diff_beta = diff_weights_data[map_index(diff_weights_d, bp.c + c)];
            norm_max = std::max(fabs(diff_beta), fabs(ref_diff_beta));
            if (norm_max < 10e-3) norm_max = data_t(1);
            EXPECT_NEAR((diff_beta - ref_diff_beta) / norm_max, 0., eps);
        }

        for (int n = 0; n < bp.mb; n++)
        for (int h = 0; h < bp.h; h++)
            for (int w = 0; w < bp.w; w++) {
                int sidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                        + h * bp.w + w;
                data_t ref_diff_src = diff_dst_data[map_index(diff_dst_d, sidx)];
                if (calculate_diff_stats) {
                        ref_diff_src -= ref_diff_beta/(bp.mb*bp.h*bp.w)
                        + (src_data[map_index(src_d, sidx)] - v_mean)
                        *ref_diff_gamma*sqrt_variance/(bp.mb*bp.h*bp.w);
                }
                ref_diff_src *= gamma*sqrt_variance;
                data_t out_diff_src = diff_src_data[map_index(diff_src_d, sidx)];
                data_t norm_max = std::max(fabs(out_diff_src), fabs(ref_diff_src));
                if (norm_max < eps) norm_max = data_t(1);
                EXPECT_NEAR((out_diff_src - ref_diff_src) / norm_max, 0., eps);
            }
    }
}

template <typename data_t>
class bnrm_test : public ::testing::TestWithParam<test_bnrm_params_t> {
private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;
    std::shared_ptr<memory> diff_src;
    std::shared_ptr<memory> diff_dst;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> diff_weights;
    std::shared_ptr<memory> mean;
    std::shared_ptr<memory> variance;
    std::shared_ptr<memory::desc> data_desc;
    std::shared_ptr<memory::desc> diff_desc;
    std::shared_ptr<batch_normalization_forward::primitive_desc> bnrm_prim_desc;
    std::shared_ptr<batch_normalization_backward::primitive_desc>
        bnrm_bwd_prim_desc;
    test_bnrm_params_t p;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        eng.reset(new engine(p.engine_kind, 0));
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_bnrm_sizes_t bs = p.sizes;
        data_desc.reset(new memory::desc({ bs.mb, bs.c, bs.h, bs.w },
                    data_type, p.formats.data_format));
        diff_desc.reset(new memory::desc({ bs.mb, bs.c, bs.h, bs.w },
                    data_type, p.formats.diff_format));

        src.reset(new memory({*data_desc, *eng}));
        dst.reset(new memory({*data_desc, *eng}));
        diff_src.reset(new memory({*diff_desc, *eng}));
        diff_dst.reset(new memory({*diff_desc, *eng}));

        auto training = prop_kind::forward_training;
        auto scoring = prop_kind::forward_scoring;


        //Forward(0u, scoring);
        //Forward(0u, training);
        //Forward(use_global_stats, training);
        //Forward(use_global_stats, scoring);
        //Forward(use_scale_shift, scoring);
        Forward(use_scale_shift, training);
        //Forward(use_scale_shift, training);
        //Forward(use_scale_shift | use_global_stats, training);

        //Backward(0u, backward_data);
        //Backward(omit_stats, backward_data);
        //Backward(use_scale_shift, backward);
        //Backward(use_scale_shift, backward_data);
        //Backward(use_scale_shift | omit_stats, backward);
        //Backward(use_scale_shift | omit_stats, backward_data);
    }

    void Forward(unsigned flags, prop_kind pk) {
        bool useScaleShift = flags & use_scale_shift;
        bool useGlobalStats = flags & use_global_stats;
        bool isTraining = pk == prop_kind::forward_training;

        auto bnrm_desc = batch_normalization_forward::desc(pk,
                    *data_desc, p.eps, flags);
        //fuse flag
        bnrm_desc.data.with_relu=p.with_relu;
        bnrm_desc.data.negative_slope=p.negative_slope;
        
        bnrm_prim_desc.reset(new batch_normalization_forward::primitive_desc(
                    bnrm_desc, *eng));

        weights.reset(new memory(bnrm_prim_desc->weights_primitive_desc()));
        if (isTraining || useGlobalStats) {
            mean.reset(new memory(bnrm_prim_desc->mean_primitive_desc()));
            variance.reset(
                    new memory(bnrm_prim_desc->variance_primitive_desc()));
        }

        fill(*src);
        if (useScaleShift) fill(*weights);
        if (useGlobalStats) {
            fill(*mean);
            fill(*variance);
        }

        auto bn = createBnrmFwd(isTraining, useGlobalStats, useScaleShift);
	
        struct timeval now,now2;	
	gettimeofday(&now, NULL);

        std::vector<primitive> pipeline;
        pipeline.push_back(bn);
        stream(stream::kind::lazy).submit(pipeline).wait();

	gettimeofday(&now2, NULL);
       double runtime=(now2.tv_sec) * 1000000 + now2.tv_usec  - ((now.tv_sec) * 1000000 + now.tv_usec );
       //fprintf(stderr, "===> time esplase %.1f us\n\n", runtime);


        check_bnrm_fwd<data_t>(p, *src, *mean, *variance, *weights, *dst, flags, pk);
    }

    void Backward(unsigned flags, prop_kind pk) {
        bool useScaleShift = flags & use_scale_shift;

        auto bnrm_bwd_desc = batch_normalization_backward::desc(
                pk, *diff_desc, *data_desc, p.eps, flags);

        bnrm_bwd_prim_desc.reset(
                new batch_normalization_backward::primitive_desc(
                bnrm_bwd_desc, *eng, *bnrm_prim_desc));

        if (useScaleShift) weights.reset(new memory(
                    bnrm_bwd_prim_desc->weights_primitive_desc()));
        diff_weights.reset(new memory(bnrm_bwd_prim_desc->diff_weights_primitive_desc()));
        mean.reset(new memory(bnrm_bwd_prim_desc->mean_primitive_desc()));
        variance.reset(new memory(
                    bnrm_bwd_prim_desc->variance_primitive_desc()));

        if (useScaleShift) fill(*weights);
        fill(*diff_dst);
        fill(*mean);
        fill(*variance);

        auto bnrm_bwd = createBnrmBwd(useScaleShift, pk);

        std::vector<primitive> pipeline;
        pipeline.push_back(bnrm_bwd);
        stream(stream::kind::lazy).submit(pipeline).wait();

        check_bnrm_bwd<data_t>(p,
                *src, *diff_dst, *mean, *variance, *weights, *diff_src, *diff_weights, flags, pk);
    }

    void fill(memory &m, data_t mean = 1.) {
        fill_data<data_t>(m.get_primitive_desc().get_size() / sizeof(data_t),
                reinterpret_cast<data_t *>(m.get_data_handle()));
    }

    primitive createBnrmFwd(bool isTraining, bool useGlobalStats,
            bool useScaleShift)
    {
        if (!isTraining && !useGlobalStats) {
            return useScaleShift
                ? batch_normalization_forward(*bnrm_prim_desc,
                    *src, *weights, *dst)
                : batch_normalization_forward(*bnrm_prim_desc, *src, *dst);
        } else {
            if (useGlobalStats) {
                return useScaleShift
                    ? batch_normalization_forward(*bnrm_prim_desc,
                        *src, (const primitive::at)*mean,
                        (const primitive::at)*variance, *weights, *dst)
                    : batch_normalization_forward(*bnrm_prim_desc,
                        *src, (const primitive::at)*mean,
                        (const primitive::at)*variance, *dst);
            } else {
                return useScaleShift
                    ? batch_normalization_forward(*bnrm_prim_desc,
                        *src, *weights, *dst, *mean, *variance)
                    : batch_normalization_forward(*bnrm_prim_desc,
                        *src, *dst, *mean, *variance);
            }
        }
    }

    primitive createBnrmBwd(bool useScaleShift, prop_kind pk)
    {
        if (useScaleShift) {
            return pk == prop_kind::backward_data
                ? batch_normalization_backward(*bnrm_bwd_prim_desc,
                    *src, *mean, *variance, *diff_dst, *weights, *diff_src)
                : batch_normalization_backward(*bnrm_bwd_prim_desc,
                    *src, *mean, *variance, *diff_dst, *weights,
                    *diff_src, *diff_weights);
        } else {
            return batch_normalization_backward(*bnrm_bwd_prim_desc,
                    *src, *mean, *variance, *diff_dst, *diff_src);
        }
    }
};

using bnrm_test_float = bnrm_test<float>;

#define EXPAND_ARGS(args) args
TEST_P(bnrm_test_float, TestsBnrm)
{
}

#define EXPAND_SIZES(mb, c, h, w) { mb, c, h, w }
#define EXPAND_FORMATS(data, diff) \
    { memory::format::data, memory::format::diff }

#define ENGINE engine::kind::cpu
#define EPS 1e-5

#define PARAMS(data, diff, mb, c, h, w, eps, fuseRelu, relu_slope) \
    test_bnrm_params_t { ENGINE, \
    EXPAND_FORMATS(data, diff), EXPAND_SIZES(mb, c, h, w), eps, fuseRelu, relu_slope }

#define PARAMS_N(...) EXPAND_ARGS(PARAMS(nchw, nchw, __VA_ARGS__))
#define PARAMS_B8(...) EXPAND_ARGS(PARAMS(nChw8c, nChw8c, __VA_ARGS__))
#define PARAMS_B16(...) EXPAND_ARGS(PARAMS(nChw16c, nChw16c, __VA_ARGS__))

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, bnrm_test_float, ::testing::Values(__VA_ARGS__))
/*
INST_TEST_CASE(Simple_NCHW,
    PARAMS_N(2, 8, 1, 1, EPS),
    PARAMS_N(2, 10, 1, 1, EPS),
    PARAMS_N(2, 8, 4, 4, EPS),
    PARAMS_N(2, 10, 4, 4, EPS)
);

INST_TEST_CASE(Simple_Blocked,
    PARAMS_B8(2, 8, 1, 1, EPS),
    PARAMS_B8(2, 8, 4, 4, EPS),
    PARAMS_B8(2, 8, 6, 6, EPS),
    PARAMS_B8(2, 16, 4, 4, EPS),
    PARAMS_B8(2, 16, 4, 4, EPS),
    PARAMS_B8(2, 16, 8, 8, EPS),
    PARAMS_B8(2, 16, 8, 8, EPS),
    PARAMS_B8(2, 16, 16, 8, EPS),
    PARAMS_B8(2, 16, 16, 8, EPS),
    PARAMS_B8(2, 16, 10, 8, EPS),
    PARAMS_B8(2, 16, 10, 8, EPS),
    PARAMS_B16(2, 16, 4, 4, EPS),
    PARAMS_B16(2, 16, 4, 4, EPS),
    PARAMS_B16(2, 16, 8, 8, EPS),
    PARAMS_B16(2, 16, 8, 8, EPS),
    PARAMS_B16(2, 16, 16, 8, EPS),
    PARAMS_B16(2, 16, 16, 8, EPS),
    PARAMS_B16(2, 16, 10, 8, EPS),
    PARAMS_B16(2, 16, 10, 8, EPS)
);

INST_TEST_CASE(GoogleNet_NCHW,
    PARAMS_N(2, 64, 112, 112, EPS),
    PARAMS_N(2, 64, 56, 56, EPS),
    PARAMS_N(2, 192, 56, 56, EPS),
    PARAMS_N(2, 96, 28, 28, EPS),
    PARAMS_N(2, 16, 28, 28, EPS),
    PARAMS_N(2, 64, 28, 28, EPS),
    PARAMS_N(2, 128, 28, 28, EPS),
    PARAMS_N(2, 32, 28, 28, EPS),
    PARAMS_N(2, 96, 28, 28, EPS),
    PARAMS_N(2, 96, 14, 14, EPS),
    PARAMS_N(2, 16, 14, 14, EPS),
    PARAMS_N(2, 192, 14, 14, EPS),
    PARAMS_N(2, 208, 14, 14, EPS),
    PARAMS_N(2, 48, 14, 14, EPS),
    PARAMS_N(2, 64, 14, 14, EPS),
    PARAMS_N(2, 112, 14, 14, EPS),
    PARAMS_N(2, 24, 14, 14, EPS),
    PARAMS_N(2, 160, 14, 14, EPS),
    PARAMS_N(2, 224, 14, 14, EPS),
    PARAMS_N(2, 128, 4, 4, EPS),
    PARAMS_N(2, 128, 14, 14, EPS),
    PARAMS_N(2, 512, 14, 14, EPS),
    PARAMS_N(2, 256, 14, 14, EPS),
    PARAMS_N(2, 144, 14, 14, EPS),
    PARAMS_N(2, 32, 14, 14, EPS),
    PARAMS_N(2, 228, 14, 14, EPS),
    PARAMS_N(2, 528, 14, 14, EPS),
    PARAMS_N(2, 320, 14, 14, EPS),
    PARAMS_N(2, 160, 7, 7, EPS),
    PARAMS_N(2, 32, 7, 7, EPS),
    PARAMS_N(2, 256, 7, 7, EPS),
    PARAMS_N(2, 320, 7, 7, EPS),
    PARAMS_N(2, 128, 7, 7, EPS),
    PARAMS_N(2, 192, 7, 7, EPS),
    PARAMS_N(2, 48, 7, 7, EPS),
    PARAMS_N(2, 384, 7, 7, EPS)
);
*/
INST_TEST_CASE(GoogleNet_Blocked_8,
    PARAMS_B8(2, 64, 112, 112, EPS,1,0),
    PARAMS_B8(2, 64, 56, 56, EPS,1,0),
    PARAMS_B8(2, 192, 56, 56, EPS,1,0),
    PARAMS_B8(2, 96, 28, 28, EPS,1,0),
    PARAMS_B8(2, 16, 28, 28, EPS,1,0),
    PARAMS_B8(2, 64, 28, 28, EPS,1,0),
    PARAMS_B8(2, 128, 28, 28, EPS,1,0),
    PARAMS_B8(2, 32, 28, 28, EPS,1,0),
    PARAMS_B8(2, 96, 28, 28, EPS,1,0),
    PARAMS_B8(2, 96, 14, 14, EPS,1,0),
    PARAMS_B8(2, 16, 14, 14, EPS,1,0),
    PARAMS_B8(2, 192, 14, 14, EPS,1,0),
    PARAMS_B8(2, 208, 14, 14, EPS,1,0),
    PARAMS_B8(2, 48, 14, 14, EPS,1,0),
    PARAMS_B8(2, 64, 14, 14, EPS,1,0),
    PARAMS_B8(2, 112, 14, 14, EPS,1,0),
    PARAMS_B8(2, 24, 14, 14, EPS,1,0),
    PARAMS_B8(2, 160, 14, 14, EPS,1,0),
    PARAMS_B8(2, 224, 14, 14, EPS,1,0),
    PARAMS_B8(2, 128, 4, 4, EPS,1,0),
    PARAMS_B8(2, 128, 14, 14, EPS,1,0),
    PARAMS_B8(2, 512, 14, 14, EPS,1,0),
    PARAMS_B8(2, 256, 14, 14, EPS,1,0),
    PARAMS_B8(2, 144, 14, 14, EPS,1,0),
    PARAMS_B8(2, 32, 14, 14, EPS,1,0),
    PARAMS_B8(2, 528, 14, 14, EPS,1,0),
    PARAMS_B8(2, 320, 14, 14, EPS,1,0),
    PARAMS_B8(2, 160, 7, 7, EPS,1,0),
    PARAMS_B8(2, 32, 7, 7, EPS,1,0),
    PARAMS_B8(2, 256, 7, 7, EPS,1,0),
    PARAMS_B8(2, 320, 7, 7, EPS,1,0),
    PARAMS_B8(2, 128, 7, 7, EPS,1,0),
    PARAMS_B8(2, 192, 7, 7, EPS,1,0),
    PARAMS_B8(2, 48, 7, 7, EPS,1,0),
    PARAMS_B8(2, 384, 7, 7, EPS,1,0)
);

INST_TEST_CASE(GoogleNet_Blocked_16,
 //   PARAMS_B16(50, 16, 112, 112, EPS),
 //   PARAMS_B16(50, 64, 112, 112, EPS),
 //   PARAMS_B16(80, 16, 112, 112, EPS),
 //    PARAMS_B16(2, 64, 7, 7, EPS)
 //   PARAMS_B16(80, 128, 112, 112, EPS),
 //  PARAMS_B16(80, 256, 112, 112, EPS),
 //   PARAMS_B16(40, 64, 112, 112, EPS),
 //   PARAMS_B16(40, 128, 112, 112, EPS),
 //   PARAMS_B16(40, 256, 112, 112, EPS),
 //   PARAMS_B16(40, 64, 56, 56, EPS),
 //   PARAMS_B16(40, 128, 56, 56, EPS),
 //   PARAMS_B16(40, 256, 56, 56, EPS),
 //   PARAMS_B16(40, 256, 56, 56, EPS)
 //   PARAMS_B16(2, 192, 56, 56, EPS)
 //
/*PARAMS_B16(50,64,112,112,EPS),
PARAMS_B16(50,256,56,56,EPS),
PARAMS_B16(50,256,56,56,EPS),
PARAMS_B16(50,256,56,56,EPS),
PARAMS_B16(50,256,56,56,EPS),
PARAMS_B16(50,512,28,28,EPS),
PARAMS_B16(50,512,28,28,EPS),
PARAMS_B16(50,512,28,28,EPS),
PARAMS_B16(50,512,28,28,EPS),
PARAMS_B16(50,512,28,28,EPS)*/

PARAMS_B16(2,64,56,56,EPS,1,0),
PARAMS_B16(50,64,56,56,EPS,1,0),
PARAMS_B16(50,64,56,56,EPS,1,0),
PARAMS_B16(50,64,56,56,EPS,1,0),
PARAMS_B16(50,64,56,56,EPS,1,0),
PARAMS_B16(50,64,56,56,EPS,1,0),
PARAMS_B16(50,128,28,28,EPS,1,0),
PARAMS_B16(50,128,28,28,EPS,1,0),
PARAMS_B16(50,128,28,28,EPS,1,0),
PARAMS_B16(50,256,14,14,EPS,1,0),
PARAMS_B16(50,256,14,14,EPS,1,0),
PARAMS_B16(50,512,7,7,EPS,1,0),
PARAMS_B16(50,512,7,7,EPS,1,0),

PARAMS_B16(2,64,56,56,EPS,0,0),
PARAMS_B16(50,64,56,56,EPS,0,0),
PARAMS_B16(50,64,56,56,EPS,0,0),
PARAMS_B16(50,64,56,56,EPS,0,0),
PARAMS_B16(50,64,56,56,EPS,0,0),
PARAMS_B16(50,64,56,56,EPS,0,0),
PARAMS_B16(50,128,28,28,EPS,0,0),
PARAMS_B16(50,128,28,28,EPS,0,0),
PARAMS_B16(50,128,28,28,EPS,0,0),
PARAMS_B16(50,256,14,14,EPS,0,0),
PARAMS_B16(50,256,14,14,EPS,0,0),
PARAMS_B16(50,512,7,7,EPS,0,0),
PARAMS_B16(50,512,7,7,EPS,0,0)

 /*   PARAMS_B16(2, 64, 112, 112, EPS),
    PARAMS_B16(2, 64, 56, 56, EPS),
    PARAMS_B16(2, 192, 56, 56, EPS),
    PARAMS_B16(2, 96, 28, 28, EPS),
    PARAMS_B16(2, 16, 28, 28, EPS),
    PARAMS_B16(2, 64, 28, 28, EPS),
    PARAMS_B16(2, 128, 28, 28, EPS),
    PARAMS_B16(2, 32, 28, 28, EPS),
    PARAMS_B16(2, 96, 28, 28, EPS),
    PARAMS_B16(2, 96, 14, 14, EPS),
    PARAMS_B16(2, 16, 14, 14, EPS),
    PARAMS_B16(2, 192, 14, 14, EPS),
    PARAMS_B16(2, 208, 14, 14, EPS),
    PARAMS_B16(2, 48, 14, 14, EPS),
    PARAMS_B16(2, 64, 14, 14, EPS),
    PARAMS_B16(2, 112, 14, 14, EPS),
    //PARAMS_B16(2, 24, 14, 14, EPS),
    PARAMS_B16(2, 160, 14, 14, EPS),
    PARAMS_B16(2, 224, 14, 14, EPS),
    PARAMS_B16(2, 128, 4, 4, EPS),
    PARAMS_B16(2, 128, 14, 14, EPS),
    PARAMS_B16(2, 512, 14, 14, EPS),
    PARAMS_B16(2, 256, 14, 14, EPS),
    PARAMS_B16(2, 144, 14, 14, EPS),
    PARAMS_B16(2, 32, 14, 14, EPS),
    PARAMS_B16(2, 528, 14, 14, EPS),
    PARAMS_B16(2, 320, 14, 14, EPS),
    PARAMS_B16(2, 160, 7, 7, EPS),
    PARAMS_B16(2, 32, 7, 7, EPS),
    PARAMS_B16(2, 256, 7, 7, EPS),
    PARAMS_B16(2, 320, 7, 7, EPS),
    PARAMS_B16(2, 128, 7, 7, EPS),
    PARAMS_B16(2, 192, 7, 7, EPS),
    PARAMS_B16(2, 48, 7, 7, EPS),
    PARAMS_B16(2, 384, 7, 7, EPS)*/
);


}
