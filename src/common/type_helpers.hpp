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

#ifndef TYPE_HELPERS_HPP
#define TYPE_HELPERS_HPP

#include <assert.h>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "mkldnn_traits.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

template <typename T>
status_t safe_ptr_assign(T * &lhs, T* rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs = rhs;
    return status::success;
}

namespace types {

inline size_t data_type_size(data_type_t data_type) {
    using namespace data_type;
    switch (data_type) {
    case f32: return sizeof(prec_traits<f32>::type);
    case s32: return sizeof(prec_traits<s32>::type);
    case s16: return sizeof(prec_traits<s16>::type);
    case s8: return sizeof(prec_traits<s8>::type);
    case u8: return sizeof(prec_traits<u8>::type);
    case data_type::undef:
    default: assert(!"unknown data_type");
    }
    return 0; /* not supposed to be reachable */
}

inline memory_format_t format_normalize(const memory_format_t fmt) {
    using namespace memory_format;
    if (utils::one_of(fmt, x, nc, nchw, nhwc, chwn, nChw8c, nChw16c, oi, io,
                oihw, ihwo, hwio, oIhw8i, oIhw16i, OIhw8i8o, OIhw16i16o,
                OIhw8i16o2i, OIhw8o16i2o, OIhw8o8i, OIhw16o16i, Oihw8o,
                Oihw16o, Ohwi8o, Ohwi16o, Ohw16oi, Ihwo16i, OhIw16o4i, Ihw16io,
                goihw, gOIhw8i8o, gOIhw16i16o, gOIhw8i16o2i, gOIhw8o16i2o,
                gOIhw8o8i, gOIhw16o16i, gOihw8o, gOihw16o, gOhwi8o, gOhwi16o,
                gOhw16oi, gIhwo16i, gIhw16io, gOhIw16o4i))
        return blocked;
    return fmt;
}

inline bool blocking_desc_is_equal(const blocking_desc_t &lhs,
        const blocking_desc_t &rhs, int ndims = TENSOR_MAX_DIMS) {
    using mkldnn::impl::utils::array_cmp;
    return lhs.offset_padding == rhs.offset_padding
        && array_cmp(lhs.block_dims, rhs.block_dims, ndims)
        && array_cmp(lhs.strides[0], rhs.strides[0], ndims)
        && array_cmp(lhs.strides[1], rhs.strides[1], ndims)
        && array_cmp(lhs.padding_dims, rhs.padding_dims, ndims)
        && array_cmp(lhs.offset_padding_to_data, rhs.offset_padding_to_data,
                ndims);
}

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    assert(lhs.primitive_kind == mkldnn::impl::primitive_kind::memory);
    assert(rhs.primitive_kind == mkldnn::impl::primitive_kind::memory);
    bool base_equal = true
        && lhs.ndims == rhs.ndims
        && mkldnn::impl::utils::array_cmp(lhs.dims, rhs.dims, lhs.ndims)
        && lhs.data_type == rhs.data_type
        && lhs.format == rhs.format; /* FIXME: normalize format? */
    if (!base_equal) return false;
    if (lhs.format == memory_format::blocked)
        return blocking_desc_is_equal(lhs.layout_desc.blocking,
                rhs.layout_desc.blocking, lhs.ndims);
    return true;
}

inline bool operator!=(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

inline memory_desc_t zero_md() {
    memory_desc_t zero{};
    zero.primitive_kind = primitive_kind::memory;
    return zero;
}

inline status_t set_default_format(memory_desc_t &md, memory_format_t fmt) {
    return mkldnn_memory_desc_init(&md, md.ndims, md.dims, md.data_type, fmt);
}

inline data_type_t default_accum_data_type(data_type_t src_dt,
        data_type_t dst_dt) {
    using namespace utils;
    using namespace data_type;

    if (one_of(f32, src_dt, dst_dt)) return f32;
    if (one_of(s16, src_dt, dst_dt)) return s32;

    if (one_of(s8, src_dt, dst_dt) || one_of(u8, src_dt, dst_dt)) return s32;

    assert(!"unimplemented use-case: no default parameters available");
    return dst_dt;
}

inline data_type_t default_accum_data_type(data_type_t src_dt,
        data_type_t wei_dt, data_type_t dst_dt, prop_kind_t prop_kind) {
    using namespace utils;
    using namespace data_type;
    using namespace prop_kind;

    /* prop_kind doesn't matter */
    if (everyone_is(f32, src_dt, wei_dt, dst_dt)) return f32;

    if (one_of(prop_kind, forward_training, forward_inference)) {
        if (src_dt == s16 && wei_dt == s16 && dst_dt == s32)
            return s32;

        if (src_dt == u8 && wei_dt == s8 && one_of(dst_dt, s32, s8, u8))
            return s32;
    } else if (prop_kind == backward_data) {
        if (src_dt == s32 && wei_dt == s16 && dst_dt == s16)
            return s32;
    } else if (prop_kind == backward_weights) {
        if (src_dt == s16 && wei_dt == s32 && dst_dt == s16)
            return s32;
    }

    assert(!"unimplemented use-case: no default parameters available");
    return dst_dt;
}

}
}
}

#include "memory_desc_wrapper.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
