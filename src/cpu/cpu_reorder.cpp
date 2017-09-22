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

#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "type_helpers.hpp"

#include "cpu/jit_reorder.hpp"
#include "cpu/simple_reorder.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using rpd_create_f = mkldnn::impl::engine_t::reorder_primitive_desc_create_f;

namespace {
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;

static const rpd_create_f cpu_reorder_impl_list[] = {
    /* fp32 <-> fp32 */
    simple_reorder_t<f32, any, f32, any, fmt_order::any, spec::direct_copy>::pd_t::create,
    simple_reorder_t<f32, any, f32, any, fmt_order::any, spec::direct_copy_except_dim_0>::pd_t::create,
    simple_reorder_t<f32, nchw, f32, nChw8c, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, nchw, f32, nChw8c, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, chwn, f32, nChw8c, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, chwn, f32, nChw8c, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, nhwc, f32, nChw8c, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, nhwc, f32, nChw8c, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, nchw, f32, nChw16c, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, nchw, f32, nChw16c, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<s32, nchw, s32, nChw16c, fmt_order::keep>::pd_t::create,
    simple_reorder_t<s32, nchw, s32, nChw16c, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, chwn, f32, nChw16c, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, chwn, f32, nChw16c, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, nhwc, f32, nChw16c, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, nhwc, f32, nChw16c, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, nChw8c, f32, nChw16c, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, nChw8c, f32, nChw16c, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, nchw, f32, nhwc, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, nchw, f32, nhwc, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, nchw, f32, chwn, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, nchw, f32, chwn, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, oihw, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, oihw, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, Ohwi8o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, Ohwi8o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, Ohwi16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, Ohwi16o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Ohwi16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Ohwi16o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Ohw16oi, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Ohw16oi, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, Ohwi16o, f32, Ohw16oi, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, Ohwi16o, f32, Ohw16oi, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, OIhw8i8o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, OIhw8i8o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, OIhw16i16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, OIhw16i16o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, OIhw8o8i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, OIhw8o8i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, OIhw16o16i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, OIhw16o16i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Oihw16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Oihw16o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, Oihw16o, f32, Ohwi16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, Oihw16o, f32, Ohwi16o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, OIhw8i8o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, OIhw8i8o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, OIhw16i16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, hwio, f32, OIhw16i16o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Ihw16io, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Ihw16io, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Ihwo16i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, oihw, f32, Ihwo16i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, Ihw16io, f32, Ihwo16i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, Ihw16io, f32, Ihwo16i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOIhw8i8o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOIhw8i8o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOIhw16i16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOIhw16i16o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOIhw8o8i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOIhw8o8i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOIhw16o16i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOIhw16o16i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOihw16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, goihw, f32, gOihw16o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, gOihw16o, f32, gOhwi16o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, gOihw16o, f32, gOhwi16o, fmt_order::reverse>::pd_t::create,
    jit_reorder_t<f32, OIhw8i8o, f32, OIhw8o8i, fmt_order::keep>::pd_t::create,
    jit_reorder_t<f32, OIhw8i8o, f32, OIhw8o8i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, OIhw8i8o, f32, OIhw8o8i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, OIhw8i8o, f32, OIhw8o8i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, OIhw16i16o, f32, OIhw16o16i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, OIhw16i16o, f32, OIhw16o16i, fmt_order::reverse>::pd_t::create,
    jit_reorder_t<f32, gOIhw8i8o, f32, gOIhw8o8i, fmt_order::keep>::pd_t::create,
    jit_reorder_t<f32, gOIhw8i8o, f32, gOIhw8o8i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, gOIhw8i8o, f32, gOIhw8o8i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, gOIhw8i8o, f32, gOIhw8o8i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, gOIhw16i16o, f32, gOIhw16o16i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<f32, gOIhw16i16o, f32, gOIhw16o16i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<f32, any, f32, any, fmt_order::any, spec::reference>::pd_t::create,
    /* s32 <-> fp32 */
    simple_reorder_t<f32, any, s32, any, fmt_order::any, spec::reference>::pd_t::create,
    simple_reorder_t<s32, any, f32, any, fmt_order::any, spec::reference>::pd_t::create,
    /* s16 <-> fp32 */
    simple_reorder_t<f32, any, s16, any, fmt_order::any, spec::reference>::pd_t::create,
    simple_reorder_t<s16, any, f32, any, fmt_order::any, spec::reference>::pd_t::create,
    /* s8 <-> fp32 */
    simple_reorder_t<f32, any, s8, any, fmt_order::any, spec::reference>::pd_t::create,
    simple_reorder_t<s8, any, f32, any, fmt_order::any, spec::reference>::pd_t::create,
    /* u8 <-> fp32 */
    simple_reorder_t<f32, any, u8, any, fmt_order::any, spec::reference>::pd_t::create,
    simple_reorder_t<u8, any, f32, any, fmt_order::any, spec::reference>::pd_t::create,
    /* s16 <-> s16 */
    simple_reorder_t<s16, any, s16, any, fmt_order::any, spec::direct_copy>::pd_t::create,
    simple_reorder_t<s16, any, s16, any, fmt_order::any, spec::direct_copy_except_dim_0>::pd_t::create,
    simple_reorder_t<s16, oihw, s16, OIhw8i16o2i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<s16, oihw, s16, OIhw8i16o2i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<s16, goihw, s16, gOIhw8i16o2i, fmt_order::keep>::pd_t::create,
    simple_reorder_t<s16, goihw, s16, gOIhw8i16o2i, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<s16, OIhw8i16o2i, s16, OIhw8o16i2o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<s16, OIhw8i16o2i, s16, OIhw8o16i2o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<s16, gOIhw8i16o2i, s16, gOIhw8o16i2o, fmt_order::keep>::pd_t::create,
    simple_reorder_t<s16, gOIhw8i16o2i, s16, gOIhw8o16i2o, fmt_order::reverse>::pd_t::create,
    simple_reorder_t<s16, any, s16, any, fmt_order::any, spec::reference>::pd_t::create,
    /* eol */
    nullptr,
};
}

const rpd_create_f *cpu_engine_t::get_reorder_implementation_list() const {
    return cpu_reorder_impl_list;
}

}
}
}
