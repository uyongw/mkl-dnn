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

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "memory_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::data_type;

status_t mkldnn_memory_desc_init(memory_desc_t *memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, memory_format_t format) {
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0 || format == memory_format::undef) {
        *memory_desc = types::zero_md();
        return success;
    }

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
        && 0 < ndims && ndims <= TENSOR_MAX_DIMS
        && one_of(data_type, f32, s32, s16, s8, u8);
    if (!args_ok) return invalid_arguments;

    memory_desc_t md;
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.primitive_kind = primitive_kind::memory;
    md.data_type = data_type;
    md.format = format;

    status_t status = success;
    switch (format) {
    case any:
        break;
    /* semidefined blocked format */
    case x:
    case nc:
    case nchw:
    case nhwc:
    case chwn:
    case nChw8c:
    case nChw16c:
    case oi:
    case io:
    case oihw:
    case ihwo:
    case hwio:
    case OIhw8i8o:
    case OIhw16i16o:
    case OIhw8i16o2i:
    case OIhw8o16i2o:
    case OIhw8o8i:
    case OIhw16o16i:
    case Oihw8o:
    case Oihw16o:
    case Ohwi8o:
    case Ohwi16o:
    case Ohw16oi:
    case Ihwo16i:
    case Ihw16io:
    case OhIw16o4i:
    case goihw:
    case gOIhw8i8o:
    case gOIhw16i16o:
    case gOIhw8i16o2i:
    case gOIhw8o16i2o:
    case gOIhw8o8i:
    case gOIhw16o16i:
    case gOihw8o:
    case gOihw16o:
    case gOhwi8o:
    case gOhwi16o:
    case gOhIw16o4i:
        status = memory_desc_wrapper::compute_blocking(md);
        break;
    /* not enough information */
    case memory_format::undef:
    case blocked:
    default:
        return invalid_arguments;
    }

    if (status == success)
        *memory_desc = md;
    return status;
}

status_t mkldnn_memory_primitive_desc_create(primitive_desc_t **memory_pd,
        const memory_desc_t *memory_desc, engine_t *engine) {
    bool args_ok = !any_null(memory_pd, memory_desc, engine)
        && memory_desc_wrapper(*memory_desc).is_defined();
    if (!args_ok) return invalid_arguments;
    return engine->memory_primitive_desc_create(
            (memory_pd_t**)memory_pd, memory_desc);
}

status_t mkldnn_view_primitive_desc_create(primitive_desc_t **view_pd,
        const primitive_desc_t *memory_pd, const dims_t dims,
        const dims_t offsets) {
    bool args_ok = !any_null(view_pd, memory_pd, dims, offsets)
        && memory_pd->kind() == primitive_kind::memory;
    if (!args_ok) return invalid_arguments;
    const memory_pd_t *mpd =
        (const memory_pd_t*)memory_pd;
    memory_desc_wrapper md(*mpd->desc());
    for (int d = 0; d < md.ndims(); ++d) {
        if (offsets[d] < 0 || (offsets[d] + dims[d] > md.dims()[d]))
            return invalid_arguments;
    }
    return memory_pd->engine()->view_primitive_desc_create(
            (view_pd_t**)view_pd, mpd, dims, offsets);
}

int mkldnn_memory_primitive_desc_equal(const primitive_desc_t *lhs,
        const primitive_desc_t *rhs) {
    bool args_ok = !any_null(lhs, rhs)
        && lhs->engine() == rhs->engine()
        && one_of(lhs->kind(), primitive_kind::memory, primitive_kind::view)
        && one_of(rhs->kind(), primitive_kind::memory, primitive_kind::view);
    if (!args_ok) return 0;
    auto l = (const memory_pd_t *)lhs;
    auto r = (const memory_pd_t *)rhs;
    /* FIXME: view! */
    return l->is_equal(r);
}

size_t mkldnn_memory_primitive_desc_get_size(const primitive_desc_t *memory_pd)
{
    bool args_ok = !any_null(memory_pd)
        && memory_pd->kind() == primitive_kind::memory;
    if (!args_ok) return 0;
    /* FIXME: view? */
    return ((memory_pd_t*)memory_pd)->get_size();
}

status_t mkldnn_memory_get_data_handle(const primitive_t *memory,
        void **handle) {
    if (any_null(handle))
        return invalid_arguments;
    if (memory == nullptr) {
        *handle = nullptr;
        return success;
    }
    if (memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    return memory->get_data_handle(handle);
}

status_t mkldnn_memory_set_data_handle(primitive_t *memory, void *handle) {
    if (any_null(memory, handle) || memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    return memory->set_data_handle(handle);
}

status_t mkldnn_concat_primitive_desc_create(primitive_desc_t **concat_pd,
        const memory_desc_t *output_d, int n, int concat_dim,
        const primitive_desc_t **input_pds) {
    bool args_ok = !any_null(concat_pd, input_pds) && n > 0;
    if (!args_ok) return invalid_arguments;
    for (int i = 0; i < n; ++i) {
        if (input_pds[i] == nullptr ||
                input_pds[i]->kind() != primitive_kind::memory)
            return invalid_arguments;
    }

    auto i_mpds = (const memory_pd_t **)input_pds;
    engine_t *engine = i_mpds[0]->engine();
    const int ndims = i_mpds[0]->desc()->ndims;
    const dims_t &dims = i_mpds[0]->desc()->dims;
    const data_type_t dt = i_mpds[0]->desc()->data_type;

    int concat_dim_sz = dims[concat_dim];
    for (int i = 1; i < n; ++i) {
        if (i_mpds[i]->engine() != engine) return invalid_arguments;
        if (i_mpds[i]->desc()->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (d == concat_dim) continue;
            if (i_mpds[i]->desc()->dims[d] != dims[d])
                return invalid_arguments;
        }
        if (i_mpds[i]->desc()->data_type != dt) return invalid_arguments;
        concat_dim_sz += i_mpds[i]->desc()->dims[concat_dim];
    }

    memory_desc_t dummy_output_d;
    if (output_d) {
        if (output_d->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (output_d->dims[d] !=
                    (d == concat_dim ? concat_dim_sz : dims[d]))
                return invalid_arguments;
        }
    } else {
        dummy_output_d = *i_mpds[0]->desc();
        dummy_output_d.dims[concat_dim] = concat_dim_sz;
        dummy_output_d.format = memory_format::any;
        output_d = &dummy_output_d;
    }

    return i_mpds[0]->engine()->concat_primitive_desc_create(
            (concat_pd_t**)concat_pd, output_d, n, concat_dim, i_mpds);
}


status_t mkldnn_sum_primitive_desc_create(primitive_desc_t **sum_pd,
        const memory_desc_t *output_d, int n, double* scale,
        const primitive_desc_t **input_pds) {
    bool args_ok = !any_null(sum_pd, input_pds, scale) && n > 0;
    if (!args_ok) return invalid_arguments;
    for (int i = 0; i < n; ++i) {
        if (input_pds[i] == nullptr ||
                input_pds[i]->kind() != primitive_kind::memory)
            return invalid_arguments;
    }

    auto i_mpds = (const memory_pd_t **)input_pds;
    engine_t *engine = i_mpds[0]->engine();
    const int ndims = i_mpds[0]->desc()->ndims;
    const dims_t &dims = i_mpds[0]->desc()->dims;
    const data_type_t dt = i_mpds[0]->desc()->data_type;

    for (int i = 1; i < n; ++i) {
        if (i_mpds[i]->engine() != engine) return invalid_arguments;
        if (i_mpds[i]->desc()->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (i_mpds[i]->desc()->dims[d] != dims[d])
                return invalid_arguments;
        }
        if (i_mpds[i]->desc()->data_type != dt) return invalid_arguments;
    }

    memory_desc_t dummy_output_d;
    if (output_d) {
        if (output_d->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (output_d->dims[d] != dims[d])
                return invalid_arguments;
        }
    } else {
        dummy_output_d = *i_mpds[0]->desc();
        dummy_output_d.format = memory_format::any;
        output_d = &dummy_output_d;
    }

    return i_mpds[0]->engine()->sum_primitive_desc_create(
            (sum_pd_t**)sum_pd, output_d, n, scale, i_mpds);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
