/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

/// @file
/// C++ API

#ifndef MKLDNN_HPP
#define MKLDNN_HPP

#include "mkldnn_config.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stdlib.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iterator>

#include "mkldnn.h"

#if MKLDNN_WITH_OPENCL
#include <CL/cl.h>
#endif

#endif

namespace mkldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_api_utils Utils
/// @{

/// A class that provides the destructor for an Intel(R) MKL-DNN C handle
template <typename T> class handle_traits {};

/// A class for wrapping an Intel(R) MKL-DNN handle. It is used as the base
/// class for primitive (#mkldnn_primitive_t), engine (#mkldnn_engine_t), and
/// stream (#mkldnn_stream_t) handles. An object of the #mkldnn::handle class
/// can be passed by value. This class enables wrapping:
///  - Newly constructed handles.
///    @n In this case, the constructed handle uses reference counting provided
///    by @p std::shared_ptr with a proper deleter function specified through
///    the @p handle_traits class.
///  - Pre-existing handles returned by the Intel(R) MKL-DNN C API (for
///    example, through mkldnn_primitive_get_primitive_desc()).
///    @n In this case, an Intel(R) MKL-DNN C API handle is wrapped without a
///    deleter because it is assumed that the handle wrapper for the original
///    object deletes the handle (this model is similar to @p std::weak_ptr).
template <typename T, typename traits=handle_traits<T>> class handle {
private:
    static mkldnn_status_t dummy_destructor(T){ return mkldnn_success; }

    std::shared_ptr<typename std::remove_pointer<T>::type> _data;
    handle(const handle &&) = delete;
    handle &operator=(const handle &&other) = delete;
protected:
    bool operator==(const T other) const { return other == _data.get(); }
    bool operator!=(const T other) const { return !(*this == other); }
public:
    /// Constructs a C handle wrapper.
    /// @param t The C handle to wrap.
    /// @param weak A flag to specify whether to construct a weak wrapper.
    handle(T t = 0, bool weak = false): _data(0) {
        reset(t, weak);
    }

    handle(const handle &other): _data(other._data) {}
    handle &operator=(const handle &other) {
        _data = other._data;
        return *this;
    }
    /// Resets the value of a C handle.
    /// @param t The new value of the C handle.
    /// @param weak A flag to specify whether the wrapper should be weak.
    void reset(T t, bool weak = false) {
        _data.reset(t, weak ? &dummy_destructor : traits::destructor);
    }

    /// Returns the value of the underlying C handle.
    T get() const { return _data.get(); }

    bool operator==(const handle &other) const { return other._data.get() == _data.get(); }
    bool operator!=(const handle &other) const { return !(*this == other); }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_memory_t> {
    static constexpr auto destructor = &mkldnn_memory_destroy;
};

template <> struct handle_traits<mkldnn_primitive_desc_t> {
    static constexpr auto destructor = &mkldnn_primitive_desc_destroy;
};

template <> struct handle_traits<mkldnn_primitive_t> {
    static constexpr auto destructor = &mkldnn_primitive_destroy;
};

template <> struct handle_traits<mkldnn_primitive_desc_iterator_t> {
    static constexpr auto destructor = &mkldnn_primitive_desc_iterator_destroy;
};
#endif

struct memory;
struct primitive_desc;

/// Base class for all computational primitives.
class primitive: public handle<mkldnn_primitive_t> {
    friend struct error;
    friend struct stream;
    using handle::handle;
public:
    /// A proxy to C primitive kind enum
    enum class kind {
        undef = mkldnn_undefined_primitive,
        reorder = mkldnn_reorder,
        concat = mkldnn_concat,
        sum = mkldnn_sum,
        convolution = mkldnn_convolution,
        deconvolution = mkldnn_deconvolution,
        shuffle = mkldnn_shuffle,
        eltwise = mkldnn_eltwise,
        softmax = mkldnn_softmax,
        pooling = mkldnn_pooling,
        lrn = mkldnn_lrn,
        batch_normalization = mkldnn_batch_normalization,
        inner_product = mkldnn_inner_product,
        rnn = mkldnn_rnn,
    };

    primitive(const_mkldnn_primitive_desc_t c_pd);
    primitive(const primitive_desc &pd);

    /// Returns the descriptor of the underlying C API primitive.
    inline const_mkldnn_primitive_desc_t get_primitive_desc() const;
    // TODO: use the C++ API wrapper structure.

    void execute(struct stream &astream,
            const std::unordered_map<int, memory> &args) const;
};

inline mkldnn_primitive_kind_t convert_to_c(primitive::kind akind) {
    return static_cast<mkldnn_primitive_kind_t>(akind);
}
/// Intel(R) MKL-DNN exception class.
///
/// This class captures the status returned by the failed C API function, error
/// message, and, optionally, handle of the primitive that caused the error.
struct error: public std::exception {
    mkldnn_status_t status;
    const char *message;

    /// Constructs an error instance.
    ///
    /// @param astatus The error status returned by the C API.
    /// @param amessage The error message.
    error(mkldnn_status_t astatus, const char *amessage)
        : status(astatus), message(amessage) {}

    /// Returns the explanatory string.
    const char *what() const noexcept override { return message; }

    /// A convenience function for wrapping calls to the C API. Checks the
    /// return status and throws an #error in case of failure.
    ///
    /// @param status The error status returned by the C API.
    /// @param message The error message.
    static void wrap_c_api(mkldnn_status_t status, const char *message) {
        if (status != mkldnn_success)
            throw error(status, message);
    }
};

const_mkldnn_primitive_desc_t primitive::get_primitive_desc() const {
    const_mkldnn_primitive_desc_t pd;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(), &pd),
            "could not get primitive descriptor by primitive");
    return pd;
}
/// @}

/// @addtogroup cpp_api_enums Common data types and enumerations
/// A proxy to @ref c_api_types in @ref c_api.
///
/// @{

enum class scratchpad_mode {
    library = mkldnn_scratchpad_mode_library,
    user = mkldnn_scratchpad_mode_user,
};

inline mkldnn_scratchpad_mode_t convert_to_c(scratchpad_mode mode) {
    return static_cast<mkldnn_scratchpad_mode_t>(mode);
}

enum class padding_kind {
    zero = mkldnn_padding_zero
};

inline mkldnn_padding_kind_t convert_to_c(padding_kind kind) {
    return static_cast<mkldnn_padding_kind_t>(kind);
}

/// Propagation kind
enum class prop_kind {
    forward_training = mkldnn_forward_training,
    forward_scoring = mkldnn_forward_scoring,
    forward_inference = mkldnn_forward_inference,
    forward = mkldnn_forward,
    backward = mkldnn_backward,
    backward_data = mkldnn_backward_data,
    backward_weights = mkldnn_backward_weights,
    backward_bias = mkldnn_backward_bias
};

inline mkldnn_prop_kind_t convert_to_c(prop_kind kind) {
    return static_cast<mkldnn_prop_kind_t>(kind);
}

enum class algorithm {
    undef = mkldnn_alg_kind_undef,
    convolution_auto = mkldnn_convolution_auto,
    convolution_direct = mkldnn_convolution_direct,
    convolution_winograd = mkldnn_convolution_winograd,
    deconvolution_direct = mkldnn_deconvolution_direct,
    deconvolution_winograd = mkldnn_deconvolution_winograd,
    eltwise_relu = mkldnn_eltwise_relu,
    eltwise_tanh = mkldnn_eltwise_tanh,
    eltwise_elu = mkldnn_eltwise_elu,
    eltwise_square = mkldnn_eltwise_square,
    eltwise_abs = mkldnn_eltwise_abs,
    eltwise_sqrt = mkldnn_eltwise_sqrt,
    eltwise_linear = mkldnn_eltwise_linear,
    eltwise_bounded_relu = mkldnn_eltwise_bounded_relu,
    eltwise_soft_relu = mkldnn_eltwise_soft_relu,
    eltwise_logistic = mkldnn_eltwise_logistic,
    lrn_across_channels = mkldnn_lrn_across_channels,
    lrn_within_channel  = mkldnn_lrn_within_channel,
    pooling_max = mkldnn_pooling_max,
    pooling_avg = mkldnn_pooling_avg,
    pooling_avg_include_padding = mkldnn_pooling_avg_include_padding,
    pooling_avg_exclude_padding = mkldnn_pooling_avg_exclude_padding,
    vanilla_rnn = mkldnn_vanilla_rnn,
    vanilla_lstm = mkldnn_vanilla_lstm,
    vanilla_gru = mkldnn_vanilla_gru,
    gru_linear_before_reset = mkldnn_gru_linear_before_reset
};

inline mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
    return static_cast<mkldnn_alg_kind_t>(aalgorithm);
}

enum class batch_normalization_flags : unsigned {
    use_global_stats = mkldnn_use_global_stats,
    use_scale_shift = mkldnn_use_scaleshift,
    fuse_bn_relu = mkldnn_fuse_bn_relu
};

inline batch_normalization_flags operator|(
        batch_normalization_flags lhs, batch_normalization_flags rhs) {
    return static_cast<batch_normalization_flags>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

inline batch_normalization_flags operator&(
        batch_normalization_flags lhs, batch_normalization_flags rhs) {
    return static_cast<batch_normalization_flags>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

inline batch_normalization_flags operator^(
        batch_normalization_flags lhs, batch_normalization_flags rhs) {
    return static_cast<batch_normalization_flags>(
            static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs));
}

inline batch_normalization_flags operator~(batch_normalization_flags rhs) {
    return static_cast<batch_normalization_flags>(~static_cast<unsigned>(rhs));
}

inline mkldnn_batch_normalization_flags_t convert_to_c(
        batch_normalization_flags aflag) {
    return static_cast<mkldnn_batch_normalization_flags_t>(aflag);
}

enum class rnn_direction {
    unidirectional_left2right = mkldnn_unidirectional_left2right,
    unidirectional_right2left = mkldnn_unidirectional_right2left,
    unidirectional = mkldnn_unidirectional,
    bidirectional_concat = mkldnn_bidirectional_concat,
    bidirectional_sum = mkldnn_bidirectional_sum,
};

inline mkldnn_rnn_direction_t convert_to_c(rnn_direction adir) {
    return static_cast<mkldnn_rnn_direction_t>(adir);
}

enum class query {
    undef = mkldnn_query_undef,

    engine = mkldnn_query_engine,
    primitive_kind = mkldnn_query_primitive_kind,

    num_of_inputs_s32 = mkldnn_query_num_of_inputs_s32,
    num_of_outputs_s32 = mkldnn_query_num_of_outputs_s32,

    time_estimate_f64 = mkldnn_query_time_estimate_f64,
    memory_consumption_s64 = mkldnn_query_memory_consumption_s64,

    scratchpad_engine = mkldnn_query_scratchpad_engine,

    impl_info_str = mkldnn_query_impl_info_str,

    op_d = mkldnn_query_op_d,
    convolution_d = mkldnn_query_convolution_d,
    deconvolution_d = mkldnn_query_deconvolution_d,
    shuffle_d = mkldnn_query_shuffle_d,
    eltwise_d = mkldnn_query_eltwise_d,
    softmax_d = mkldnn_query_softmax_d,
    pooling_d = mkldnn_query_pooling_d,
    lrn_d = mkldnn_query_lrn_d,
    batch_normalization_d = mkldnn_query_batch_normalization_d,
    inner_product_d = mkldnn_query_inner_product_d,
    rnn_d = mkldnn_query_rnn_d,

    src_md = mkldnn_query_src_md,
    diff_src_md = mkldnn_query_diff_src_md,
    weights_md = mkldnn_query_weights_md,
    diff_weights_md = mkldnn_query_diff_weights_md,
    dst_md = mkldnn_query_dst_md,
    diff_dst_md = mkldnn_query_diff_dst_md,
    workspace_md = mkldnn_query_workspace_md,
    scratchpad_md = mkldnn_query_scratchpad_md,
};

inline mkldnn_query_t convert_to_c(query aquery) {
    return static_cast<mkldnn_query_t>(aquery);
}

/// Backend kinds
enum class backend_kind {
    /// Native backend
    native = mkldnn_backend_native,
    /// OpenCL backend
    ocl = mkldnn_backend_ocl,
};

inline mkldnn_backend_kind_t convert_to_c(backend_kind akind) {
    return static_cast<mkldnn_backend_kind_t>(akind);
}

/// @}

/// @addtogroup cpp_api_attr Attributes
/// An extension for controlling primitive behavior.
///
/// @sa @ref c_api_attributes in @ref c_api
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_post_ops_t> {
    static constexpr auto destructor = &mkldnn_post_ops_destroy;
};
#endif

/// Post operations
struct post_ops: public handle<mkldnn_post_ops_t> {

    /// Creates an empty sequence of post operations.
    post_ops() {
        mkldnn_post_ops_t result;
        error::wrap_c_api(mkldnn_post_ops_create(&result),
                "could not create post operation sequence");
        reset(result);
    }

    int len() const { return mkldnn_post_ops_len(get()); }

    primitive::kind kind(int index) const {
        error::wrap_c_api(
                index < len() ? mkldnn_success : mkldnn_invalid_arguments,
                "post_ops index is out of range");
        return static_cast<primitive::kind>(mkldnn_post_ops_get_kind(get(),
                    index));
    }

    /// Appends accumulation (sum) post operation. Prior to accumulating the
    /// result, the previous value would be multiplied by @p scale.
    ///
    /// The kind of this post operation is #mkldnn_sum.
    ///
    /// This feature might improve performance for cases like residual learning
    /// blocks, where the result of convolution is accumulated to the previously
    /// computed activations. The parameter @p scale might be extreme for the
    /// integer-based computations when the result and previous activations have
    /// different logical scaling factors.
    ///
    /// In the simplest case when the accumulation is the only post operation,
    /// the computations would be:
    /// dst[] <- scale * dst[] + op(...) // instead of dst[] <- op(...)
    ///
    /// @note
    ///     This post operation (as well as all the others) disregards the
    ///     original layout of the destination; that is, the layout of the
    ///     original destination is expected to be the same as the layout of the
    ///     stored destination.
    void append_sum(float scale = 1.) {
        error::wrap_c_api(mkldnn_post_ops_append_sum(get(), scale),
                "could not append sum");
    }

    /// Gets the parameters of the accumulation (sum) post operation with index
    /// @p index.
    void get_params_sum(int index, float &scale) const {
        error::wrap_c_api(mkldnn_post_ops_get_params_sum(get(), index, &scale),
                "could not get sum params");
    }

    /// Appends eltwise post operation.
    ///
    /// The kind of this post operation is #mkldnn_eltwise.
    ///
    /// In the simplest case when the eltwise is the only post operation, the
    /// computations would be:
    /// dst[] <- scale * eltwise_op ( op(...) ) // instead of dst[] <- op(...)
    /// where eltwise_op is configured with the given parameters.
    void append_eltwise(float scale, algorithm alg, float alpha,
            float beta) {
        error::wrap_c_api(mkldnn_post_ops_append_eltwise(get(), scale,
                    convert_to_c(alg), alpha, beta),
                "could not append eltwise");
    }

    /// Gets the eltwise parameters of the post operation with index @p index.
    void get_params_eltwise(int index, float &scale, algorithm &alg,
            float &alpha, float &beta) const {
        mkldnn_alg_kind_t c_alg;
        error::wrap_c_api(mkldnn_post_ops_get_params_eltwise(get(), index,
                    &scale, &c_alg, &alpha, &beta),
                "could not get eltwise params");
        alg = static_cast<algorithm>(c_alg);
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_primitive_attr_t> {
    static constexpr auto destructor = &mkldnn_primitive_attr_destroy;
};
#endif

/// Primitive attributes
struct primitive_attr: public handle<mkldnn_primitive_attr_t> {
    primitive_attr() {
        mkldnn_primitive_attr_t result;
        error::wrap_c_api(mkldnn_primitive_attr_create(&result),
                "could not create a primitive attr");
        reset(result);
    }

    scratchpad_mode get_scratchpad_mode() const {
        mkldnn_scratchpad_mode_t result;
        error::wrap_c_api(mkldnn_primitive_attr_get_scratchpad_mode(
                    get(), &result), "could not get scratchpad mode");
        return scratchpad_mode(result);
    }

    void set_scratchpad_mode(scratchpad_mode mode) {
        error::wrap_c_api(mkldnn_primitive_attr_set_scratchpad_mode(
                    get(), mkldnn::convert_to_c(mode)),
                "could not set scratchpad mode");
    }

    /// Gets correspondence scale @p mask and a constant floating point vector
    /// of output @p scales previously set by set_output_scales.
    void get_output_scales(int &mask, std::vector<float> &scales) const
    {
        mkldnn_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(mkldnn_primitive_attr_get_output_scales(get(),
                    &count, &c_mask, &c_scales),
                "could not get int output scales");
        scales.resize(count);

        mask = c_mask;
        for (mkldnn_dim_t c = 0; c < count; ++c)
            scales[c] = c_scales[c];
    }

    /// Sets output scales for primitive operations. The correspondence scale
    /// @p mask is stored for future use.
    ///
    /// The @p mask argument defines the correspondence between the output
    /// tensor dimensions and the @p scales vector. Set the i-th bit of @p mask
    /// to 1 to use a dedicated scaling factor for each slice of the output
    /// tensor over the i-th dimension. Set @p mask to 0 to use a common
    /// scaling factor for the whole output tensor.
    ///
    /// @note
    ///      The dimension order is always native and does not depend on the
    ///      actual layout used. Examples:
    ///       - 2D dimensional data the order of dimensions is always: (n, c)
    ///       - 4D dimensional data the order is always: (n, c, h, w)
    ///       - 5D dimensional weights the order is always: (g, oc, ic, kh, kw)
    void set_output_scales(int mask, const std::vector<float> &scales)
    {
        error::wrap_c_api(mkldnn_primitive_attr_set_output_scales(get(),
                    (mkldnn_dim_t)scales.size(), mask, &scales[0]),
                "could not set int output scales");
    }

    /// Returns @p post_ops previously set by set_post_ops.
    const post_ops get_post_ops() const {
        post_ops result;
        const_mkldnn_post_ops_t c_result;
        error::wrap_c_api(mkldnn_primitive_attr_get_post_ops(get(), &c_result),
                "could not get post operation sequence");
        result.reset(const_cast<mkldnn_post_ops_t>(c_result), true);
        return result;
    }
    
    /// Sets @p post_ops for future use.
    void set_post_ops(post_ops ops) {
        error::wrap_c_api(mkldnn_primitive_attr_set_post_ops(get(), ops.get()),
                "could not set post operation sequence");
    }

    /// Sets quantization @p scale and @p shift for RNN data tensors.  For
    /// performance reasons, the low-precision configuration of the RNN
    /// primitive expects input activations to have the unsigned int8 data type.
    /// Scale and shift used to quantize floating-point data to unsigned integer
    /// must be passed to the RNN primitive using attributes.
    /// @note
    ///     Quantization scale and shift are common for src_layer, src_iter,
    ///     dst_iter, and dst_layer.
    void set_rnn_data_qparams(const float scale, const float shift)
    {
        error::wrap_c_api(mkldnn_primitive_attr_set_rnn_data_qparams(get(),
                    scale, shift), "could not set rnn data int scale/shift");
    }

    /// Sets quantization scales @p weights_scales for RNN weights tensors.  The
    /// low-precision configuration of the RNN primitive expects input weights
    /// to have the signed int8 data type. Scales used to quantize
    /// floating-point data to signed integer must be passed to the RNN
    /// primitive using attributes.  The @p mask argument defines correspondence
    /// between output tensor dimensions and the @p weights_scales array. Set
    /// the i-th bit of @p mask to 1 to use a dedicated scaling factor for each
    /// slice of the output tensor over the i-th dimension. Set @p mask to 0 to
    /// use a common scaling factor for the whole output tensor.
    /// @note
    ///      The dimension order is always native and does not depend on the
    ///      actual layout used. For example, five-dimensional weights always
    ///      have (l, d, i, g, o) logical dimension ordering.
    /// @note
    ///     Quantization scales are common for weights_layer and
    ///     weights_iteration
    /// @note
    ///     There is no way to check whether @p count corresponds to @p mask
    ///     until an actual primitive descriptor is created, so it is the user's
    ///     responsibility to set proper values. The following formula must
    ///     hold:
    ///
    ///      \f[count = \prod\limits_{d \in mask} output.dims[d]\f]
    void set_rnn_weights_qparams(int mask, const std::vector<float> &scales)
    {
        error::wrap_c_api(mkldnn_primitive_attr_set_rnn_weights_qparams(get(),
                    (int)scales.size(), mask, &scales[0]),
                "could not set rnn weights int scales");
    }
};

/// @}

/// @addtogroup cpp_api_engine Engine
/// Engine operations.
///
/// @sa @ref c_api_engine in @ref c_api
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_engine_t> {
    static constexpr auto destructor = &mkldnn_engine_destroy;
};
#endif

/// An execution engine.
struct engine: public handle<mkldnn_engine_t> {
    friend class primitive;
    // gcc bug??? using handle::handle;

    /// Kinds of engines.
    enum class kind {
        /// An unspecified engine
        any = mkldnn_any_engine,
        /// CPU engine
        cpu = mkldnn_cpu,
        /// GPU engine
        gpu = mkldnn_gpu,
    };

    /// Returns the number of engines of a certain kind.
    ///
    /// @param akind The kind of engines to count.
    static size_t get_count(kind akind) {
        return mkldnn_engine_get_count(convert_to_c(akind));
    }

    /// Constructs an engine.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index The index of the engine. Must be less than the value
    ///              returned by #get_count() for this particular kind
    ///              of engine.
    engine(kind akind, size_t index) {
        mkldnn_engine_t aengine;
        error::wrap_c_api(
                mkldnn_engine_create(&aengine,
                    convert_to_c(akind), index),
                "could not create an engine");
        reset(aengine);
    }

#if MKLDNN_WITH_OPENCL
    engine(kind akind, cl_device_id device, cl_context context) {
        mkldnn_engine_t aengine;
        error::wrap_c_api(mkldnn_engine_create_ocl(&aengine,
                                  convert_to_c(akind), device, context),
                "could not create an engine");
        reset(aengine);
    }
#endif

    explicit engine(const mkldnn_engine_t& aengine)
        : handle(aengine, true) {}

    engine(const handle<mkldnn_primitive_desc_t> &pd) {
        mkldnn_engine_t engine_q;
        error::wrap_c_api(
                mkldnn_primitive_desc_query(pd.get(),
                    mkldnn::convert_to_c(mkldnn::query::engine), 0, &engine_q),
                "could not get engine from primitive_desc");
        reset(engine_q, true);
    }

    kind get_kind() const {
        mkldnn_engine_kind_t akind;
        error::wrap_c_api(mkldnn_engine_get_kind(get(), &akind),
                "could not get the engine kind");
        return static_cast<engine::kind>(akind);
    }

    backend_kind get_backend_kind() const {
        mkldnn_backend_kind_t abackend_kind;
        error::wrap_c_api(mkldnn_engine_get_backend_kind(get(), &abackend_kind),
                "could not get the backend kind of the engine");
        return static_cast<backend_kind>(abackend_kind);
    }

#if MKLDNN_WITH_OPENCL
    cl_context get_ocl_context() const {
        cl_context context = nullptr;
        error::wrap_c_api(mkldnn_engine_get_ocl_context(get(), &context),
                "could not get a context handle");
        return context;
    }

    cl_device_id get_ocl_device() const {
        cl_device_id device = nullptr;
        error::wrap_c_api(mkldnn_engine_get_ocl_device(get(), &device),
                "could not get a device handle");
        return device;
    }
#endif


    template <class primitive_desc>
    static engine query(const primitive_desc &pd) {
        mkldnn_engine_t engine_q;
        error::wrap_c_api(
                mkldnn_primitive_desc_query(pd.get(),
                    mkldnn::convert_to_c(mkldnn::query::engine), 0, &engine_q),
                "could not get engine from primitive_desc");

        return engine(engine_q);
    }

private:
    static mkldnn_engine_kind_t convert_to_c(kind akind) {
        return static_cast<mkldnn_engine_kind_t>(akind);
    }
};

/// @}

/// @addtogroup cpp_api_stream Stream
/// Execution stream operations
///
/// @sa @ref c_api_stream in @ref c_api
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_stream_t> {
    static constexpr auto destructor = &mkldnn_stream_destroy;
};
#endif

/// An execution stream.
struct stream: public handle<mkldnn_stream_t> {
    using handle::handle;

    enum class flags : unsigned {
        default_order = mkldnn_stream_default_order,
        in_order = mkldnn_stream_default_order,
        out_of_order = mkldnn_stream_out_of_order,
        default_flags = mkldnn_stream_default_flags,
    };

    /// Constructs a stream.
    stream(const engine &aengine,
            flags aflags = flags::default_flags) {
        mkldnn_stream_t astream;
        error::wrap_c_api(mkldnn_stream_create(&astream, aengine.get(),
                                  static_cast<mkldnn_stream_flags_t>(aflags)),
                "could not create a stream");
        reset(astream);
    }

#if MKLDNN_WITH_OPENCL
    stream(const engine &eng, cl_command_queue queue) {
        mkldnn_stream_t astream;
        error::wrap_c_api(mkldnn_stream_create_ocl(&astream, eng.get(), queue),
                "could not create a stream");
        reset(astream);
    }

    cl_command_queue get_ocl_command_queue() const {
        cl_command_queue queue = nullptr;
        error::wrap_c_api(mkldnn_stream_get_ocl_command_queue(get(), &queue),
                "could not get OpenCL command queue");
        return queue;
    }
#endif


    /// Waits for all primitives in the stream to finish.
    stream &wait() {
        error::wrap_c_api(mkldnn_stream_wait(get()),
               "could not wait a stream");
        return *this;
    }
};

inline stream::flags operator|(stream::flags lhs, stream::flags rhs) {
    return static_cast<stream::flags>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

inline stream::flags operator&(stream::flags lhs, stream::flags rhs) {
    return static_cast<stream::flags>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

inline stream::flags operator^(stream::flags lhs, stream::flags rhs) {
    return static_cast<stream::flags>(
            static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs));
}

inline stream::flags operator~(stream::flags rhs) {
    return static_cast<stream::flags>(~static_cast<unsigned>(rhs));
}

/// @}

/// @addtogroup cpp_api_memory_related Memory and memory related operations
/// @{

/// @addtogroup cpp_api_memory Memory
/// A primitive to describe and store data.
///
/// For more information, refer to @ref c_api_memory in @ref c_api.
/// @{

/// Memory that describes the data.
struct memory: public handle<mkldnn_memory_t> {
    public:
    typedef mkldnn_dim_t dim;
    typedef std::vector<dim> dims;

    template <typename T> static void validate_dims(const std::vector<T> &v) {
        if (v.size() > MKLDNN_MAX_NDIMS)
            throw error(mkldnn_invalid_arguments, "invalid dimensions");
    }

    /// Data type specification. See #mkldnn_data_type_t for a detailed
    /// description.
    enum class data_type {
        undef = mkldnn_data_type_undef,
        f16 = mkldnn_f16,
        f32 = mkldnn_f32,
        s32 = mkldnn_s32,
        s8 = mkldnn_s8,
        u8 = mkldnn_u8,
    };

    /// Memory format tag specification. See #mkldnn_format_tag_t for a
    /// detailed description.
    enum class format_tag {
        undef = mkldnn_format_tag_undef,
        /// Placeholder memory format tag. The primitive selects a format
        /// automatically.
        any = mkldnn_format_tag_any,
        a = mkldnn_a,
        ab = mkldnn_ab,
        abc = mkldnn_abc,
        abcd = mkldnn_abcd,
        abcde = mkldnn_abcde,
        abcdef = mkldnn_abcdef,
        abdec = mkldnn_abdec,
        acb = mkldnn_acb,
        acbde = mkldnn_acbde,
        acdb = mkldnn_acdb,
        acdeb = mkldnn_acdeb,
        ba = mkldnn_ba,
        bac = mkldnn_bac,
        bacd = mkldnn_bacd,
        bcda = mkldnn_bcda,
        cba = mkldnn_cba,
        cdba = mkldnn_cdba,
        cdeba = mkldnn_cdeba,
        decab = mkldnn_decab,
        Abc16a = mkldnn_Abc16a,
        ABc16a16b = mkldnn_ABc16a16b,
        aBc16b = mkldnn_aBc16b,
        ABc16b16a = mkldnn_ABc16b16a,
        Abc4a = mkldnn_Abc4a,
        aBc4b = mkldnn_aBc4b,
        ABc4b16a4b = mkldnn_ABc4b16a4b,
        ABc4b4a = mkldnn_ABc4b4a,
        ABc8a16b2a = mkldnn_ABc8a16b2a,
        ABc8a8b = mkldnn_ABc8a8b,
        aBc8b = mkldnn_aBc8b,
        ABc8b16a2b = mkldnn_ABc8b16a2b,
        ABc8b8a = mkldnn_ABc8b8a,
        Abcd16a = mkldnn_Abcd16a,
        ABcd16a16b = mkldnn_ABcd16a16b,
        aBcd16b = mkldnn_aBcd16b,
        ABcd16b16a = mkldnn_ABcd16b16a,
        aBCd16b16c = mkldnn_aBCd16b16c,
        aBCd16c16b = mkldnn_aBCd16c16b,
        Abcd4a = mkldnn_Abcd4a,
        aBcd4b = mkldnn_aBcd4b,
        ABcd4b16a4b = mkldnn_ABcd4b16a4b,
        ABcd4b4a = mkldnn_ABcd4b4a,
        aBCd4c16b4c = mkldnn_aBCd4c16b4c,
        aBCd4c4b = mkldnn_aBCd4c4b,
        ABcd8a16b2a = mkldnn_ABcd8a16b2a,
        ABcd8a8b = mkldnn_ABcd8a8b,
        aBcd8b = mkldnn_aBcd8b,
        ABcd8b16a2b = mkldnn_ABcd8b16a2b,
        aBCd8b16c2b = mkldnn_aBCd8b16c2b,
        ABcd8b8a = mkldnn_ABcd8b8a,
        aBCd8b8c = mkldnn_aBCd8b8c,
        aBCd8c16b2c = mkldnn_aBCd8c16b2c,
        aBCd8c8b = mkldnn_aBCd8c8b,
        Abcde16a = mkldnn_Abcde16a,
        ABcde16a16b = mkldnn_ABcde16a16b,
        aBcde16b = mkldnn_aBcde16b,
        ABcde16b16a = mkldnn_ABcde16b16a,
        aBCde16b16c = mkldnn_aBCde16b16c,
        aBCde16c16b = mkldnn_aBCde16c16b,
        aBCde2c8b4c = mkldnn_aBCde2c8b4c,
        Abcde4a = mkldnn_Abcde4a,
        aBcde4b = mkldnn_aBcde4b,
        ABcde4b4a = mkldnn_ABcde4b4a,
        aBCde4b4c = mkldnn_aBCde4b4c,
        aBCde4c16b4c = mkldnn_aBCde4c16b4c,
        aBCde4c4b = mkldnn_aBCde4c4b,
        Abcde8a = mkldnn_Abcde8a,
        ABcde8a8b = mkldnn_ABcde8a8b,
        aBcde8b = mkldnn_aBcde8b,
        ABcde8b16a2b = mkldnn_ABcde8b16a2b,
        aBCde8b16c2b = mkldnn_aBCde8b16c2b,
        ABcde8b8a = mkldnn_ABcde8b8a,
        aBCde8b8c = mkldnn_aBCde8b8c,
        ABcd4a8b8a4b = mkldnn_ABcd4a8b8a4b,
        ABcd2a8b8a2b = mkldnn_ABcd2a8b8a2b,
        aBCde4b8c8b4c = mkldnn_aBCde4b8c8b4c,
        aBCde2b8c8b2c = mkldnn_aBCde2b8c8b2c,
        aBCde8c16b2c = mkldnn_aBCde8c16b2c,
        aBCde8c8b = mkldnn_aBCde8c8b,
        aBcdef16b = mkldnn_aBcdef16b,
        aBCdef16b16c = mkldnn_aBCdef16b16c,
        aBCdef16c16b = mkldnn_aBCdef16c16b,
        aBcdef4b = mkldnn_aBcdef4b,
        aBCdef4c4b = mkldnn_aBCdef4c4b,
        aBCdef8b8c = mkldnn_aBCdef8b8c,
        aBCdef8c16b2c = mkldnn_aBCdef8c16b2c,
        aBCdef8c8b = mkldnn_aBCdef8c8b,
        aBdc16b = mkldnn_aBdc16b,
        aBdc4b = mkldnn_aBdc4b,
        aBdc8b = mkldnn_aBdc8b,
        aBdec16b = mkldnn_aBdec16b,
        aBdec4b = mkldnn_aBdec4b,
        aBdec8b = mkldnn_aBdec8b,
        aBdefc16b = mkldnn_aBdefc16b,
        aCBdef16c16b = mkldnn_aCBdef16c16b,
        aBdefc4b = mkldnn_aBdefc4b,
        aBdefc8b = mkldnn_aBdefc8b,
        Acb16a = mkldnn_Acb16a,
        Acb4a = mkldnn_Acb4a,
        Acb8a = mkldnn_Acb8a,
        aCBd16b16c = mkldnn_aCBd16b16c,
        aCBde16b16c = mkldnn_aCBde16b16c,
        aCBde16c16b = mkldnn_aCBde16c16b,
        Acdb16a = mkldnn_Acdb16a,
        Acdb4a = mkldnn_Acdb4a,
        Acdb8a = mkldnn_Acdb8a,
        Acdeb16a = mkldnn_Acdeb16a,
        Acdeb4a = mkldnn_Acdeb4a,
        Acdeb8a = mkldnn_Acdeb8a,
        BAc16a16b = mkldnn_BAc16a16b,
        BAcd16a16b = mkldnn_BAcd16a16b,
        BAcd16b16a = mkldnn_BAcd16b16a,
        ABcd32a32b = mkldnn_ABcd32a32b,
        BAcde16b16 = mkldnn_BAcde16b16a,
        aBdec32b   = mkldnn_aBdec32b,
        Abcdef16a  = mkldnn_Abcdef16a,
        aCBde16c16 = mkldnn_aCBde16c16b,
        Acdb32a    = mkldnn_Acdb32a,
        format_tag_last = mkldnn_format_tag_last,

        x = mkldnn_x,
        nc = mkldnn_nc,
        cn = mkldnn_cn,
        ncw = mkldnn_ncw,
        nwc = mkldnn_nwc,
        nchw = mkldnn_nchw,
        nhwc = mkldnn_nhwc,
        chwn = mkldnn_chwn,
        ncdhw = mkldnn_ncdhw,
        ndhwc = mkldnn_ndhwc,
        oi = mkldnn_oi,
        io = mkldnn_io,
        oiw = mkldnn_oiw,
        wio = mkldnn_wio,
        oihw = mkldnn_oihw,
        hwio = mkldnn_hwio,
        ihwo = mkldnn_ihwo,
        iohw = mkldnn_iohw,
        oidhw = mkldnn_oidhw,
        dhwio = mkldnn_dhwio,
        goiw = mkldnn_goiw,
        goihw = mkldnn_goihw,
        hwigo = mkldnn_hwigo,
        giohw = mkldnn_giohw,
        goidhw = mkldnn_goidhw,
        tnc = mkldnn_tnc,
        ntc = mkldnn_ntc,
        ldsnc = mkldnn_ldsnc,
        ldigo = mkldnn_ldigo,
        ldgoi = mkldnn_ldgoi,
        ldgo = mkldnn_ldgo,
        nCdhw16c = mkldnn_nCdhw16c,
        nCdhw4c = mkldnn_nCdhw4c,
        nCdhw8c = mkldnn_nCdhw8c,
        nChw16c = mkldnn_nChw16c,
        nChw4c = mkldnn_nChw4c,
        nChw8c = mkldnn_nChw8c,
        nCw16c = mkldnn_nCw16c,
        nCw4c = mkldnn_nCw4c,
        nCw8c = mkldnn_nCw8c,
        NChw16n16c = mkldnn_NChw16n16c,
        NCdhw16n16c = mkldnn_NCdhw16n16c,
        NChw32n32c  = mkldnn_NChw32n32c,
        IOhw16i16o  = mkldnn_IOhw16i16o,
        Ohwi32o     = mkldnn_Ohwi32o,
        IOdhw16i16o = mkldnn_IOdhw16i16o,
        gIOhw16i16o = mkldnn_gIOhw16i16o,
        gOhwi32o    = mkldnn_gOhwi32o,
        Goidhw16g   = mkldnn_Goidhw16g,
        IOw16o16i = mkldnn_IOw16o16i,
        OIw16i16o = mkldnn_OIw16i16o,
        OIw16o16i = mkldnn_OIw16o16i,
        Oiw16o = mkldnn_Oiw16o,
        OIw4i16o4i = mkldnn_OIw4i16o4i,
        OIw4i4o = mkldnn_OIw4i4o,
        Oiw4o = mkldnn_Oiw4o,
        OIw8i16o2i = mkldnn_OIw8i16o2i,
        OIw8i8o = mkldnn_OIw8i8o,
        OIw8o16i2o = mkldnn_OIw8o16i2o,
        OIw8o8i = mkldnn_OIw8o8i,
        Owi16o = mkldnn_Owi16o,
        Owi4o = mkldnn_Owi4o,
        Owi8o = mkldnn_Owi8o,
        IOhw16o16i = mkldnn_IOhw16o16i,
        Ohwi16o = mkldnn_Ohwi16o,
        Ohwi4o = mkldnn_Ohwi4o,
        Ohwi8o = mkldnn_Ohwi8o,
        OIhw16i16o = mkldnn_OIhw16i16o,
        OIhw16o16i = mkldnn_OIhw16o16i,
        Oihw16o = mkldnn_Oihw16o,
        OIhw4i16o4i = mkldnn_OIhw4i16o4i,
        OIhw4i4o = mkldnn_OIhw4i4o,
        Oihw4o = mkldnn_Oihw4o,
        OIhw8i16o2i = mkldnn_OIhw8i16o2i,
        OIhw8i8o = mkldnn_OIhw8i8o,
        OIhw8o16i2o = mkldnn_OIhw8o16i2o,
        OIhw8o8i = mkldnn_OIhw8o8i,
        Odhwi16o = mkldnn_Odhwi16o,
        Odhwi4o = mkldnn_Odhwi4o,
        Odhwi8o = mkldnn_Odhwi8o,
        OIdhw16i16o = mkldnn_OIdhw16i16o,
        OIdhw16o16i = mkldnn_OIdhw16o16i,
        Oidhw16o = mkldnn_Oidhw16o,
        OIdhw4i4o = mkldnn_OIdhw4i4o,
        Oidhw4o = mkldnn_Oidhw4o,
        OIdhw8i16o2i = mkldnn_OIdhw8i16o2i,
        OIdhw8i8o = mkldnn_OIdhw8i8o,
        OIdhw8o8i = mkldnn_OIdhw8o8i,
        gIOw16o16i = mkldnn_gIOw16o16i,
        gOIw16i16o = mkldnn_gOIw16i16o,
        gOIw16o16i = mkldnn_gOIw16o16i,
        gOiw16o = mkldnn_gOiw16o,
        gOIw4i16o4i = mkldnn_gOIw4i16o4i,
        gOIw4i4o = mkldnn_gOIw4i4o,
        gOiw4o = mkldnn_gOiw4o,
        gOIw8i16o2i = mkldnn_gOIw8i16o2i,
        gOIw8i8o = mkldnn_gOIw8i8o,
        gOIw8o16i2o = mkldnn_gOIw8o16i2o,
        gOIw8o8i = mkldnn_gOIw8o8i,
        gOwi16o = mkldnn_gOwi16o,
        gOwi4o = mkldnn_gOwi4o,
        gOwi8o = mkldnn_gOwi8o,
        gIOhw16o16i = mkldnn_gIOhw16o16i,
        gOhwi16o = mkldnn_gOhwi16o,
        gOhwi4o = mkldnn_gOhwi4o,
        gOhwi8o = mkldnn_gOhwi8o,
        Goihw16g = mkldnn_Goihw16g,
        gOIhw16i16o = mkldnn_gOIhw16i16o,
        gOIhw16o16i = mkldnn_gOIhw16o16i,
        gOihw16o = mkldnn_gOihw16o,
        gOIhw2i8o4i = mkldnn_gOIhw2i8o4i,
        gOIhw4i16o4i = mkldnn_gOIhw4i16o4i,
        gOIhw4i4o = mkldnn_gOIhw4i4o,
        gOIhw4o4i = mkldnn_gOIhw4o4i,
        gOihw4o = mkldnn_gOihw4o,
        Goihw8g = mkldnn_Goihw8g,
        gOIhw8i16o2i = mkldnn_gOIhw8i16o2i,
        gOIhw8i8o = mkldnn_gOIhw8i8o,
        gOIhw8o16i2o = mkldnn_gOIhw8o16i2o,
        OIhw4o8i8o4i = mkldnn_OIhw4o8i8o4i,
        OIhw2o8i8o2i = mkldnn_OIhw2o8i8o2i,
        gOIhw4o8i8o4i = mkldnn_gOIhw4o8i8o4i,
        gOIhw2o8i8o2i = mkldnn_gOIhw2o8i8o2i,
        gOIhw8o8i = mkldnn_gOIhw8o8i,
        gIOdhw16i16o = mkldnn_gIOdhw16i16o,
        gOdhwi16o = mkldnn_gOdhwi16o,
        gOdhwi4o = mkldnn_gOdhwi4o,
        gOdhwi8o = mkldnn_gOdhwi8o,
        gOIdhw16i16o = mkldnn_gOIdhw16i16o,
        gOIdhw16o16i = mkldnn_gOIdhw16o16i,
        gOidhw16o = mkldnn_gOidhw16o,
        gOIdhw4i4o = mkldnn_gOIdhw4i4o,
        gOidhw4o = mkldnn_gOidhw4o,
        gOIdhw8i16o2i = mkldnn_gOIdhw8i16o2i,
        gOIdhw8i8o = mkldnn_gOIdhw8i8o,
        gOIdhw8o8i = mkldnn_gOIdhw8o8i,
    };

    /// A memory descriptor.
    struct desc {
        friend struct memory;
        /// The underlying C API data structure.
        mkldnn_memory_desc_t data;

        /// Constructs a zero memory descriptor
        desc(): data() {}

        /// Constructs a memory descriptor.
        ///
        /// @param adims Data dimensions
        /// @param adata_type Data precision/type.
        /// @param aformat Data layout format tag.
        desc(const dims &adims, data_type adata_type,
                format_tag aformat) {
            validate_dims(adims);
            error::wrap_c_api(mkldnn_memory_desc_init_by_tag(&data, (int)adims.size(),
                        adims.size() == 0 ? nullptr : &adims[0],
                        convert_to_c(adata_type), convert_to_c(aformat)),
                    "could not initialize a memory descriptor");
        }

        /// Constructs a memory descriptor from a C API data structure.
        ///
        /// @param adata A C API #mkldnn_memory_desc_t structure.
        desc(const mkldnn_memory_desc_t &adata): data(adata) {}

        /// Constructs a sub-memory descriptor.
        //
        /// @param adims Sizes of a sub-memory
        /// @param offsets Offsets of a sub-memory
        desc submemory_desc(const dims &adims, const dims &offsets) {
            mkldnn_memory_desc_t sub_md;
            error::wrap_c_api(mkldnn_memory_desc_init_submemory(&sub_md,
                        &data, &adims[0], &offsets[0]),
                    "could not initialize a sub-memory");
            return desc(sub_md);
        }

        /// Returns the number of bytes required to allocate the memory
        /// described including the padding area.
        size_t get_size() const { return mkldnn_memory_desc_get_size(&data); }

        bool operator==(const desc &other) const {
            return mkldnn_memory_desc_equal(&data, &other.data) != 0;
        }

        bool operator!=(const desc &other) const { return !operator==(other); }
    };

    /// Constructs a memory.
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine.
    /// @param ahandle handle.
    memory(const desc &md, const engine &aengine, void *ahandle) {
        mkldnn_memory_t result;
        error::wrap_c_api(mkldnn_memory_create(&result, &md.data,
                    aengine.get(), ahandle), "could not create a memory");
        reset(result);
    }

    /// Constructs a memory.
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine.
    memory(const desc &md, const engine &aengine)
        : memory(md, aengine, MKLDNN_MEMORY_ALLOCATE) {}

    /// Returns the descriptor of the memory.
    desc get_desc() const {
        const mkldnn_memory_desc_t *cdesc;
        error::wrap_c_api(mkldnn_memory_get_memory_desc(get(), &cdesc),
                "could not get memory descriptor from a memory");
        return desc(*cdesc);
    }

    /// Returns the engine of the memory.
    engine get_engine() const {
        mkldnn_engine_t engine_q;
        error::wrap_c_api(mkldnn_memory_get_engine(get(), &engine_q),
                "could not get engine from a memory");
        return engine(engine_q);
    }

    /// Returns a handle of the data contained in the memory.
    ///
    /// On the CPU engine, this is a pointer to the allocated memory.
    void *get_data_handle() const {
        void *handle;
        error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle),
                "could not get native handle");
        return handle;
    }

    void set_data_handle(void *handle) const {
        error::wrap_c_api(mkldnn_memory_set_data_handle(get(), handle),
                "could not set native handle");
    }

    /// Maps the data of the memory.
    ///
    /// Mapping allows to read/write directly from/to the memory contents for
    /// backends that do not support direct accessing.
    ///
    /// Mapping is an exclusive operation - a memory object cannot be used in
    /// other operations until this memory object is unmapped.
    /// @tparam T Type of the pointer to be mapped.
    template <typename T = void>
    T *map_data() const {
        void *mapped_ptr;
        error::wrap_c_api(mkldnn_memory_map_data(get(), &mapped_ptr),
                "could not map the data");
        return static_cast<T *>(mapped_ptr);
    }

    /// Unmaps the previously mapped data for the memory.
    ///
    /// Any changes of the mapped data are synchronized back to the memory
    /// after the call is complete. The mapped pointer must be
    /// obtained through a map_data() call.
    void unmap_data(void *mapped_ptr) const {
        error::wrap_c_api(mkldnn_memory_unmap_data(get(), mapped_ptr),
                "could not unmap the data");
    }

#if MKLDNN_WITH_OPENCL
    cl_mem get_ocl_mem_object() const {
        cl_mem mem_object;
        error::wrap_c_api(mkldnn_memory_get_ocl_mem_object(get(), &mem_object),
                "could not get OpenCL memory object");
        return mem_object;
    }

    void set_ocl_mem_object(cl_mem mem_object) {
        error::wrap_c_api(mkldnn_memory_set_ocl_mem_object(get(), mem_object),
                "could not set OpenCL memory object");
    }
#endif

    // Must go away or be private:
    static mkldnn_data_type_t convert_to_c(data_type adata_type) {
        return static_cast<mkldnn_data_type_t>(adata_type);
    }
    static mkldnn_format_tag_t convert_to_c(format_tag aformat) {
        return static_cast<mkldnn_format_tag_t>(aformat);
    }
};

inline bool operator==(mkldnn_data_type_t a, memory::data_type b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(mkldnn_data_type_t a, memory::data_type b) {
    return !(a == b);
}
inline bool operator==(memory::data_type a, mkldnn_data_type_t b) {
    return b == a;
}
inline bool operator!=(memory::data_type a, mkldnn_data_type_t b) {
    return !(a == b);
}

inline bool operator==(mkldnn_format_tag_t a, memory::format_tag b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(mkldnn_format_tag_t a, memory::format_tag b) {
    return !(a == b);
}
inline bool operator==(memory::format_tag a, mkldnn_format_tag_t b) {
    return b == a;
}
inline bool operator!=(memory::format_tag a, mkldnn_format_tag_t b) {
    return !(a == b);
}

/// @}

/// @addtogroup cpp_api_reorder Reorder
/// A primitive to copy data between memory formats.
///
/// @sa @ref dev_guide_reorder in developer guide
/// @sa @ref c_api_reorder in @ref c_api
/// @{

/// Initializes a reorder primitive using the description of the source
/// (@p src_engine and @p src_md) and destination (@p dst_engine and @p dst_md)
/// memory, and an @p attr attribute.
struct reorder : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const engine &src_engine, const memory::desc &src_md,
                const engine &dst_engine, const memory::desc &dst_md,
                const primitive_attr &aattr = primitive_attr()) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_reorder_primitive_desc_create(&result,
                        &src_md.data, src_engine.get(),
                        &dst_md.data, dst_engine.get(), aattr.get()),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }

        primitive_desc(const memory &src, const memory &dst,
                const primitive_attr &aattr = primitive_attr()) {
            mkldnn_primitive_desc_t result;
            auto src_md = src.get_desc();
            auto dst_md = dst.get_desc();
            error::wrap_c_api(mkldnn_reorder_primitive_desc_create(&result,
                        &src_md.data, src.get_engine().get(),
                        &dst_md.data, dst.get_engine().get(), aattr.get()),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }

        memory::desc scratchpad_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(query::scratchpad_md), 0);
            if (cdesc == nullptr)
                return memory::desc();
            return memory::desc(*cdesc);
        }

        engine scratchpad_engine() {
            mkldnn_engine_t engine_q;
            error::wrap_c_api(
                mkldnn_primitive_desc_query(get(),
                    mkldnn::convert_to_c(query::scratchpad_engine), 0, &engine_q),
                "could not get scratchpad engine from reorder primitive_desc");

            return engine(engine_q);
        }

        engine get_engine() { return engine::query(*this); }
    };

    reorder(const primitive_desc &pd): primitive(pd.get()) {}

    reorder(const memory &src, const memory &dst):
        primitive(primitive_desc(src, dst).get()) {}

    void execute(stream astream, memory &src, memory &dst) {
        primitive::execute(astream,
                {{MKLDNN_ARG_FROM, src}, {MKLDNN_ARG_TO, dst}});
    }
};

/// @}

/// @addtogroup cpp_api_concat Concat
/// A primitive to concatenate data by arbitrary dimension.
///
/// @sa @ref dev_guide_concat in developer guide
/// @sa @ref c_api_concat in @ref c_api
/// @{

/// Implements primitive descriptor and primitive for concat.
///
/// Creates an out-of-place primitive descriptor for concatenation of @p n
/// inputs by @p concat_dimension with resulting @p output_desc memory
/// descriptor. @p output_desc can be NULL or specified with the
/// #mkldnn::memory::format_tag::any format kind--in this case, the appropriate memory
/// format would be chosen automatically.
struct concat : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        std::vector<mkldnn_memory_desc_t> cpp_to_c(
                const std::vector<memory::desc> &srcs) {
            std::vector<mkldnn_memory_desc_t> c_api_srcs;
            c_api_srcs.reserve(srcs.size());
            for (const auto &s : srcs) c_api_srcs.push_back(s.data);
            return c_api_srcs;
        }

        primitive_desc(const memory::desc &dst, int concat_dimension,
                const std::vector<memory::desc> &srcs, const engine &aengine,
                const primitive_attr &aattr = primitive_attr()) {
            auto c_api_srcs = cpp_to_c(srcs);

            mkldnn_primitive_desc_t result;
            error::wrap_c_api(
                    mkldnn_concat_primitive_desc_create(&result, &dst.data,
                            (int)c_api_srcs.size(), concat_dimension,
                            &c_api_srcs[0], aattr.get(), aengine.get()),
                    "could not create a concat primitive descriptor");
            reset(result);
        }

        primitive_desc(int concat_dimension,
                const std::vector<memory::desc> &srcs, const engine &aengine,
                const primitive_attr &aattr = primitive_attr()) {
            auto c_api_srcs = cpp_to_c(srcs);

            mkldnn_primitive_desc_t result;
            error::wrap_c_api(
                    mkldnn_concat_primitive_desc_create(&result, nullptr,
                            (int)c_api_srcs.size(), concat_dimension,
                            &c_api_srcs[0], aattr.get(), aengine.get()),
                    "could not create a concat primitive descriptor");
            reset(result);
        }

        memory::desc dst_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(query::dst_md), 0);
            error::wrap_c_api(
                    cdesc == nullptr ? mkldnn_runtime_error : mkldnn_success,
                    "could not get a dst memory descriptor");
            return memory::desc(*cdesc);
        }

        memory::desc scratchpad_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(query::scratchpad_md), 0);
            if (cdesc == nullptr)
                return memory::desc();
            return memory::desc(*cdesc);
        }

        engine get_engine() { return engine::query(*this); }
    };

    concat(const primitive_desc &pd): primitive(pd.get()) {}
};

/// @}

/// @addtogroup cpp_api_sum Sum
/// A primitive to sum data.
///
/// @sa @ref dev_guide_sum in developer guide
/// @sa @ref c_api_sum in @ref c_api
/// @{

/// Creates an out-of-place sum primitive descriptor for sum of @p n inputs
/// multiplied by the scale with resulting @p output_desc memory descriptor.
/// @p output_desc can be NULL or specified with the
/// #mkldnn::memory::format_tag::any format kind--in this case, the
/// appropriate memory format would be chosen automatically.
struct sum : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        std::vector<mkldnn_memory_desc_t> cpp_to_c(
                const std::vector<memory::desc> &srcs) {
            std::vector<mkldnn_memory_desc_t> c_api_srcs;
            c_api_srcs.reserve(srcs.size());
            for (const auto &s : srcs) c_api_srcs.push_back(s.data);
            return c_api_srcs;
        }

        primitive_desc(const memory::desc &dst,
                const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs, const engine &aengine,
                const primitive_attr &aattr = primitive_attr()) {
            error::wrap_c_api(scales.size() == srcs.size()
                    ? mkldnn_success : mkldnn_invalid_arguments,
                "number of scales not equal to number of srcs");

            auto c_api_srcs = cpp_to_c(srcs);

            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_sum_primitive_desc_create(
                    &result, &dst.data, (int)c_api_srcs.size(),
                    &scales[0], &c_api_srcs[0], aattr.get(), aengine.get()),
                "could not create a sum primitive descriptor");
            reset(result);
        }

        primitive_desc(const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs, const engine &aengine,
                const primitive_attr &aattr = primitive_attr()) {
            error::wrap_c_api(scales.size() == srcs.size()
                    ? mkldnn_success : mkldnn_invalid_arguments,
                "number of scales not equal to number of srcs");

            auto c_api_srcs = cpp_to_c(srcs);
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_sum_primitive_desc_create(&result,
                        nullptr, (int)c_api_srcs.size(), &scales[0],
                        &c_api_srcs[0], aattr.get(), aengine.get()),
                    "could not create a sum primitive descriptor");
            reset(result);
        }

        memory::desc dst_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(query::dst_md), 0);
            error::wrap_c_api(
                    cdesc == nullptr ? mkldnn_runtime_error : mkldnn_success,
                    "could not get a dst memory descriptor");
            return memory::desc(*cdesc);
        }

        memory::desc scratchpad_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(query::scratchpad_md), 0);
            if (cdesc == nullptr)
                return memory::desc();
            return memory::desc(*cdesc);
        }

        engine get_engine() { return engine::query(*this); }
    };

    sum(const primitive_desc &pd): primitive(pd.get()) {}
};

/// @}

/// @}

/// @addtogroup cpp_api_primitives Primitives
/// @{

/// @addtogroup cpp_api_primitive_descriptors Primitive descriptors
/// @{

/// A base class for all primitive descriptors.
struct primitive_desc : public handle<mkldnn_primitive_desc_t> {

    /// Creates a primitive descriptor from given @p op_desc, @p attr, @p
    /// engine, and optionally a hint primitive descriptor from forward
    /// propagation.
    primitive_desc(const_mkldnn_op_desc_t desc, const primitive_attr *attr,
            const engine &e, const_mkldnn_primitive_desc_t hint_fwd_pd) {
        mkldnn_primitive_desc_iterator_t iterator = nullptr;
        mkldnn_status_t status = mkldnn_primitive_desc_iterator_create(
                &iterator, desc, attr ? attr->get() : nullptr, e.get(),
                hint_fwd_pd);
        error::wrap_c_api(status,
                "could not create a primitive descriptor iterator");
        pd_iterator.reset(iterator);
        fetch_impl();
    }

    engine get_engine() { return engine::query(*this); }

    primitive_attr get_primitive_attr() const {
        const_mkldnn_primitive_attr_t const_cattr;
        error::wrap_c_api(mkldnn_primitive_desc_get_attr(get(), &const_cattr),
                "could not get attributes");
        mkldnn_primitive_attr_t cattr;
        error::wrap_c_api(mkldnn_primitive_attr_clone(&cattr, const_cattr),
                "could not clone attributes");

        primitive_attr attr;
        attr.reset(cattr);
        return attr;
    }

    /// Returns implementation name
    const char *impl_info_str() const {
        const char *res;
        error::wrap_c_api(mkldnn_primitive_desc_query(get(),
                    mkldnn_query_impl_info_str, 0, &res),
                "could not query implementation info string");
        return res;
    }

    /// Queries the memory::dim value (same as int64_t)
    memory::dim query_s64(query q) const {
        memory::dim res;
        mkldnn_status_t status = mkldnn_primitive_desc_query(get(),
                mkldnn::convert_to_c(q), 0, &res);
        return status == mkldnn_success ? res : 0;
    }

    /// Advances the next implementation for the given op descriptor.
    ///
    /// Returns:
    /// - @c true on success
    /// - @c false if the last implementation reached, and
    ///   the primitive descriptor itself is kept unchanged
    bool next_impl() {
        mkldnn_status_t status = mkldnn_primitive_desc_iterator_next(
                pd_iterator.get());
        if (status == mkldnn_iterator_ends) return false;
        error::wrap_c_api(status, "primitive descriptor iterator next failed");

        fetch_impl();
        return true;
    }

    /// Queries and returns requested memory descriptor.
    memory::desc query_md(query what, int idx = 0) const {
        std::vector<query> valid_q{ query::src_md, query::diff_src_md,
            query::weights_md, query::diff_weights_md, query::dst_md,
            query::diff_dst_md, query::workspace_md, query::scratchpad_md };
        if (!std::any_of(valid_q.cbegin(), valid_q.cend(),
                    [=](query q) { return what == q; }))
            throw error(mkldnn_invalid_arguments, "invalid memory query");

        const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                get(), mkldnn::convert_to_c(what), idx);
        if (cdesc == nullptr) return memory::desc();

        return memory::desc(*cdesc);
    }

    // register specialized queries, e.g. src_desc()
#   define REG_QUERY_MD(name, what, idx) \
    memory::desc name ## _desc() const { return query_md(query::what ## _md, idx); }

  private:
    handle<mkldnn_primitive_desc_iterator_t> pd_iterator;
    void fetch_impl() {
        mkldnn_primitive_desc_t pd = mkldnn_primitive_desc_iterator_fetch(
                pd_iterator.get());
        error::wrap_c_api(pd != nullptr ? mkldnn_success : mkldnn_runtime_error,
                "could not fetch a primitive descriptor from the iterator");
        reset(pd);
    }
};

/// @}

/// @addtogroup cpp_api_convolution Convolution
/// Computes a forward propagation, backward propagation, or weight update
/// for convolution operation with bias on a batch of multi-dimensional tensors.
///
/// @sa @ref dev_guide_convolution in developer guide
/// @sa @ref c_api_convolution in @ref c_api
/// @{

/// Convolution forward propagation.
///
/// Implements descriptor, primitive descriptor, and primitive
/// for the convolution forward propagation.
struct convolution_forward: public primitive {

    /// Descriptor for convolution forward propagation.
    struct desc {
        mkldnn_convolution_desc_t data;

        /// Initializes a descriptor for convolution forward propagation without
        /// bias using @p aprop_kind (possible values are
        /// #mkldnn::forward_training and #mkldnn::forward_inference),
        /// @p aalgorithm, memory descriptors, @p strides, @p padding_l,
        /// @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution forward descriptor");
        }

        /// Initializes a descriptor for convolution forward propagation with
        /// bias using @p prop_kind (possible values are
        /// #mkldnn::forward_training and #mkldnn::forward_inference), @p
        /// aalgorithm, memory descriptors, @p strides, @p padding_l, @p
        /// padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution forward descriptor");
        }

        /// Initializes a descriptor for dilated convolution forward propagation
        /// without bias using @p prop_kind (possible values are
        /// #mkldnn::forward_training and #mkldnn::forward_inference),
        /// @p aalgorithm, memory descriptors, @p strides, @p dilates,
        /// @p padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                mkldnn_dilated_convolution_forward_desc_init(&data,
                    mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &dilates[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated convolution forward descriptor");
        }

        /// Initializes a descriptor for dilated convolution forward propagation
        /// with bias using @p prop_kind (possible values are
        /// #mkldnn::forward_training and #mkldnn::forward_inference),
        /// @p aalgorithm, memory descriptors, @p strides, @p dilates,
        /// @p padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                mkldnn_dilated_convolution_forward_desc_init(&data,
                    mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &dilates[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated convolution forward descriptor");
        }
    };

    /// Primitive descriptor for convolution forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {

        /// Initializes a primitive descriptor for convolution forward
        /// propagation.
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        /// Initializes a primitive descriptor for convolution forward
        /// propagation with attributes defined by @p attr.
        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(bias, weights, 1);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    /// Creates a convolution forward propagation primitive from the
    /// corresponding primitive descriptor.
    convolution_forward(const primitive_desc &pd): primitive(pd) {}
};

/// Convolution backward propagation.
///
/// Implements descriptor, primitive descriptor, and primitive for the
/// convolution backward propagation.
struct convolution_backward_data : public primitive {

    /// Descriptor for convolution backward propagation.
    struct desc {
        mkldnn_convolution_desc_t data;

        /// Initializes a descriptor for convolution backward propagation
        /// using @p aalgorithm, memory descriptors, @p strides, @p
        /// padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_data_desc_init(
                        &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward data descriptor");
        }

        /// Initializes a descriptor for dilated convolution backward
        /// propagation using @p aalgorithm, memory descriptors, @p strides, @p
        /// padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                mkldnn_dilated_convolution_backward_data_desc_init(
                    &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                    &weights_desc.data, &diff_dst_desc.data,
                    &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                    mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward data descriptor");
        }
    };

    /// Primitive descriptor for convolution backward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {

        /// Initializes primitive descriptor for convolution backward
        /// propagation.
        primitive_desc(const desc &desc, const engine &e,
                const convolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        /// Initializes primitive descriptor for convolution backward
        /// propagation with attributes defined by @p attr.
        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const convolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    /// Creates a convolution backward propagation primitive from the
    /// corresponding primitive descriptor.
    convolution_backward_data(const primitive_desc &pd): primitive(pd) {}
};

/// Convolution weight update.
///
/// Implements descriptor, primitive descriptor, and primitive for the
/// convolution weight update.
struct convolution_backward_weights : public primitive {

    /// Descriptor for convolution weight update.
    struct desc {
        mkldnn_convolution_desc_t data;

        /// Initializes a descriptor for convolution weight update with bias
        /// using @p aalgorithm, memory descriptors, @p strides, @p padding_l,
        /// @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }

        /// Initializes a descriptor for convolution weight update without
        /// bias using @p aalgorithm, memory descriptors, @p strides, @p
        /// padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }

        /// Initializes a descriptor for dilated convolution weight update
        /// with bias using @p aalgorithm, memory descriptors, @p strides,
        /// @p dilates @p padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }

        /// Initializes a descriptor for dilated convolution weight update
        /// without bias using @p aalgorithm, memory descriptors, @p strides,
        /// @p dilates @p padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &dilates[0],  &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }

    };

    /// Primitive descriptor for convolution weight update.
    struct primitive_desc : public mkldnn::primitive_desc {

        /// Initializes a primitive descriptor for convolution weight update.
        primitive_desc(const desc &desc, const engine &e,
                const convolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        /// Initializes a primitive descriptor for convolution weight update
        /// with attributes defined by @p attr.
        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const convolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(diff_weights, diff_weights, 0);
        REG_QUERY_MD(diff_bias, diff_weights, 1);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    /// Creates convolution weight update primitive from corresponding
    /// primitive descriptor.
    convolution_backward_weights(const primitive_desc &pd): primitive(pd) {}
};

/// @}
//
/// @addtogroup cpp_api_deconvolution Deconvolution
/// A primitive to compute deconvolution using different algorithms.
///
/// @sa @ref c_api_deconvolution in @ref c_api
/// @{

/// Deconvolution forward propagation.
///
/// Implements descriptor, primitive descriptor, and primitive
/// for the deconvolution forward propagation.
struct deconvolution_forward: public primitive {

    /// Descriptor for convolution forward propagation.
    struct desc {
        mkldnn_deconvolution_desc_t data;

        /// Initializes a descriptor for deconvolution forward propagation
        /// with bias using @p prop_kind (possible values are
        /// #mkldnn::forward_training and #mkldnn::forward_inference), @p
        /// aalgorithm, memory descriptors, @p strides, @p padding_l, @p
        /// padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution forward descriptor");
        }

        /// Initializes a descriptor for deconvolution forward propagation
        /// without bias using @p prop_kind (possible values are
        /// #mkldnn::forward_training and #mkldnn::forward_inference), @p
        /// aalgorithm, memory descriptors, @p strides, @p padding_l, @p
        /// padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution forward descriptor");
        }

        /// Initializes a descriptor for dilated deconvolution forward
        /// propagation with bias using @p aprop_kind (possible values are
        /// #mkldnn::forward_training and #mkldnn::forward_inference), @p
        /// aalgorithm memory descriptors, @p strides, @p dilates, @p
        /// padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &dilates[0], &padding_l[0],
                        &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated deconvolution forward descriptor");
        }

        /// Initializes a descriptor for dilated deconvolution forward
        /// propagation without bias using @p aprop_kind (possible values are
        /// #mkldnn::forward_training and #mkldnn::forward_inference), @p
        /// aalgorithm, memory descriptors, @p strides, @p dilates, @p
        /// padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &dilates[0], &padding_l[0],
                        &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated deconvolution forward descriptor");
        }
    };

     /// Primitive descriptor for deconvolution forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {

        /// Initializes a primitive descriptor for deconvolution forward
        /// propagation.
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        /// Initializes primitive descriptor for deconvolution forward
        /// propagation with attributes defined by @p attr.
        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(bias, weights, 1);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    /// Creates a deconvolution forward propagation primitive from the
    /// corresponding primitive descriptor.
    deconvolution_forward(const primitive_desc &pd): primitive(pd) {}
};

/// Deconvolution backward propagation.
///
/// Implements descriptor, primitive descriptor, and primitive for the
/// deconvolution backward propagation.
struct deconvolution_backward_data : public primitive {

    /// Descriptor for deconvolution backward propagation.
    struct desc {
        mkldnn_deconvolution_desc_t data;

        /// Initializes a descriptor for deconvolution backward propagation
        /// using @p aalgorithm, memory descriptors, @p strides, @p
        /// padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_backward_data_desc_init(
                        &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution backward data descriptor");
        }

        /// Initializes descriptor for dilated deconvolution backward propagation
        /// using @p aalgorithm, memory descriptors, @p strides, @p
        /// padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_backward_data_desc_init(
                        &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated deconvolution backward data descriptor");
        }
    };

    /// Primitive descriptor for deconvolution backward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {

        /// Initializes a primitive descriptor for deconvolution backward
        /// propagation.
        primitive_desc(const desc &desc, const engine &e,
                const deconvolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        /// Initializes a primitive descriptor for deconvolution backward
        /// propagation with attributes defined by @p attr.
        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const deconvolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    /// Creates a deconvolution backward propagation primitive from the
    /// corresponding primitive descriptor.
    deconvolution_backward_data(const primitive_desc &pd): primitive(pd) {}
};

/// Deconvolution weight update.
///
/// Implements descriptor, primitive descriptor, and primitive
/// for the deconvolution weight update.
struct deconvolution_backward_weights : public primitive {

     /// Descriptor for deconvolution weight update.
    struct desc {
        mkldnn_deconvolution_desc_t data;

        /// Initializes a descriptor for deconvolution weight update with bias
        /// using @p aalgorithm, memory descriptors, @p strides, @p padding_l,
        /// @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution backward weights descriptor");
        }

        /// Initializes a descriptor for deconvolution weight update without
        /// bias using @p aalgorithm, memory descriptors, @p strides, @p
        /// padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution backward weights descriptor");
        }

        /// Initializes a descriptor for dilated deconvolution weight update
        /// with bias using @p aalgorithm, memory descriptors, @p strides, @p
        /// dilates @p padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated  deconvolution backward weights descriptor");
        }

        /// Initializes a descriptor for dilated deconvolution weight update
        /// without bias using @p aalgorithm, memory descriptors, @p strides,
        /// @p dilates @p padding_l, @p padding_r, and @p apadding_kind.
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated deconvolution backward weights descriptor");
        }
    };

    /// Primitive descriptor for deconvolution weight update.
    struct primitive_desc : public mkldnn::primitive_desc {

        /// Initializes a primitive descriptor for deconvolution weight update.
        primitive_desc(const desc &desc, const engine &e,
                const deconvolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        /// Initializes a primitive descriptor for deconvolution weight update
        /// with attributes defined by @p attr.
        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const deconvolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(diff_weights, diff_weights, 0);
        REG_QUERY_MD(diff_bias, diff_weights, 1);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    /// Creates a deconvolution weight update primitive from the corresponding
    /// primitive descriptor.
    deconvolution_backward_weights(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_lrn LRN
/// A primitive to perform local response normalization (LRN) across or within
/// channels.
///
/// @sa @ref dev_guide_lrn in developer guide
/// @sa @ref c_api_lrn in @ref c_api
/// @{

/// Local response normalization for forward propagation. Implements
/// descriptor, primitive descriptor, and primitive.
struct lrn_forward : public primitive {

    /// Descriptor for local response normalization forward propagation.
    struct desc {
        mkldnn_lrn_desc_t data;

        /// Initializes a descriptor for forward propagation using @p prop_kind
        /// (possible values are #mkldnn::forward_training and
        /// #mkldnn::forward_inference), @p aalgorithm, memory descriptor @p
        /// data_desc, and regularization parameters @p local_size, @p alpha, @p
        /// beta, and @p k.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, memory::dim local_size,
                float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(mkldnn_lrn_forward_desc_init(&data,
                mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, local_size, alpha, beta, k),
                "could not create a lrn forward descriptor");
        }
    };

    /// Primitive descriptor for local response normalization forward
    /// propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    lrn_forward(const primitive_desc &pd): primitive(pd) {}
};

/// Local response normalization for backward propagation.  Implements
/// descriptor, primitive descriptor, and primitive.
struct lrn_backward : public primitive {

    /// Descriptor for local response normalization backward propagation.
    struct desc {
        mkldnn_lrn_desc_t data;

        /// Initializes a descriptor for backward propagation using @p aalgorithm,
        /// memory descriptors @p data_desc and @p diff_data_desc, and
        /// regularization parameters @p local_size, @p alpha, @p beta, and
        /// @p k.
        desc(algorithm aalgorithm, const memory::desc &data_desc,
                const memory::desc &diff_data_desc, memory::dim local_size,
                float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(mkldnn_lrn_backward_desc_init(&data,
                convert_to_c(aalgorithm), &diff_data_desc.data,
                &data_desc.data, local_size, alpha, beta, k),
                "could not create a lrn backward descriptor");
        }
    };

    /// Primitive descriptor for local response normalization backward
    /// propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const lrn_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const lrn_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    lrn_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_pooling Pooling
/// A primitive to perform max or average pooling.
///
/// @sa @ref dev_guide_pooling in developer guide
/// @sa @ref c_api_pooling in @ref c_api
/// @{

/// Pooling for forward propagation.  Implements descriptor, primitive
/// descriptor, and primitive.
struct pooling_forward : public primitive {

    /// Descriptor for pooling forward propagation.
    struct desc {
        mkldnn_pooling_desc_t data;

        /// Initializes a pooling descriptor for forward propagation using @p
        /// aprop_kind (possible values are #mkldnn::forward_training and
        /// #mkldnn::forward_inference), @p aalgorithm, memory descriptors, and
        /// pooling parameters in the spatial domain: @p strides, @p kernel
        /// sizes, @p padding_l, @p padding_r, and @p apadding_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims kernel,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_pooling_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm),
                        &src_desc.data, &dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a forward pooling descriptor");
        }
    };

    /// Primitive descriptor for pooling forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    pooling_forward(const primitive_desc &pd): primitive(pd) {}
};

struct pooling_backward : public primitive {

    /// Descriptor for pooling backward propagation.
    struct desc {
        mkldnn_pooling_desc_t data;

        /// Initializes a pooling descriptor for backward propagation using @p
        /// aalgorithm, memory descriptors, and pooling parameters in the spatial
        /// domain: @p strides, @p kernel sizes, @p padding_l, @p padding_r,
        /// and @p apadding_kind.
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims &strides,
                const memory::dims &kernel,
                const memory::dims &padding_l,
                const memory::dims &padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_pooling_backward_desc_init(&data,
                        convert_to_c(aalgorithm),
                        &diff_src_desc.data, &diff_dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a backward pooling descriptor");
        }
    };

    /// Primitive descriptor for pooling backward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const pooling_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const pooling_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    pooling_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_eltwise Eltwise
/// A primitive to compute element-wise operations such as rectified linear
/// unit (ReLU).
///
/// Both forward and backward passes support in-place operation; that is, src
/// and dst point to the same memory for forward pass, and diff_dst and
/// diff_src point to the same memory for backward pass.
///
/// @warning Because the original src is required for backward pass, in-place
/// forward pass in general cannot be applied during training. However, for
/// some kinds of element-wise operations (namely ReLU with alpha parameter
/// equals 0), dst and src can be interchangeable for the backward pass, which
/// enables performance of in-place forward even for training.
///
/// @sa @ref dev_guide_eltwise in developer guide
/// @sa @ref c_api_eltwise in @ref c_api
/// @{

/// Element-wise operations for forward propagation.  Implements descriptor,
/// primitive descriptor, and primitive.
struct eltwise_forward : public primitive {

    /// Initializes an eltwise descriptor for forward propagation using @p
    /// prop_kind (possible values are #mkldnn::forward_training and
    /// #mkldnn::forward_inference), @p aalgorithm algorithm, memory
    /// descriptor @p data_desc, @p alpha, and @p beta parameters.
    struct desc {
        mkldnn_eltwise_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, T alpha = 0, T beta = 0) {
            error::wrap_c_api(mkldnn_eltwise_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        mkldnn::convert_to_c(aalgorithm), &src_desc.data,
                        static_cast<float>(alpha), static_cast<float>(beta)),
                    "could not create a eltwise forward descriptor");
        }
    };

    /// Primitive descriptor for eltwise forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    eltwise_forward(const primitive_desc &pd): primitive(pd) {}
};

/// Element-wise operations for backward propagation.  Implements descriptor,
/// primitive descriptor, and primitive.
struct eltwise_backward : public primitive {

    /// Initializes an eltwise descriptor for backward propagation using @p
    /// aalgorithm algorithm memory descriptors @p diff_data_desc and @p
    /// data_desc, and the @p alpha and @p beta parameters.
    struct desc {
        mkldnn_eltwise_desc_t data;

        template <typename T>
        desc(algorithm aalgorithm, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, T alpha = 0, T beta = 0) {
            error::wrap_c_api(mkldnn_eltwise_backward_desc_init(&data,
                        mkldnn::convert_to_c(aalgorithm), &diff_data_desc.data,
                        &data_desc.data, static_cast<float>(alpha),
                        static_cast<float>(beta)),
                    "could not create a eltwise backward descriptor");
        }
    };

    /// Primitive descriptor for eltwise backward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const eltwise_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const eltwise_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    eltwise_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_softmax Softmax
/// A primitive to perform softmax.
///
/// @sa @ref dev_guide_softmax in developer guide
/// @sa @ref c_api_softmax in @ref c_api
/// @{

/// Softmax for forward propagation.  Implements descriptor, primitive
/// descriptor, and primitive.
struct softmax_forward : public primitive {

    /// Descriptor for softmax forward propagation.
    struct desc {
        mkldnn_softmax_desc_t data;

        /// Initializes a softmax descriptor for forward propagation using @p
        /// prop_kind (possible values are #mkldnn::forward_training and
        /// #mkldnn::forward_inference) and memory descriptor @p data_desc.
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             int softmax_axis) {
            error::wrap_c_api(mkldnn_softmax_forward_desc_init(&data,
                    mkldnn::convert_to_c(aprop_kind), &data_desc.data,
                    softmax_axis),
                "could not create a softmax forward descriptor");
        }
    };

    /// Primitive descriptor for softmax forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    softmax_forward(const primitive_desc &pd): primitive(pd) {}
};

/// Softmax for backward propagation.  Implements descriptor, primitive
/// descriptor, and primitive.
struct softmax_backward : public primitive {

    /// Descriptor for softmax backward propagation.
    struct desc {
        mkldnn_softmax_desc_t data;

        /// Initializes a softmax descriptor for backward propagation using
        /// memory descriptors @p diff_desc and @p data_desc.
        desc(const memory::desc &diff_desc, const memory::desc &data_desc,
                int softmax_axis) {
            error::wrap_c_api(mkldnn_softmax_backward_desc_init(&data,
                        &diff_desc.data, &data_desc.data, softmax_axis),
                    "could not init a backward softmax descriptor");
        }
    };

    /// Primitive descriptor for softmax backward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const softmax_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const softmax_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    softmax_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_batch_normalization Batch normalization
/// A primitive to perform batch normalization.
///
/// Both forward and backward passes support in-place operation; that is, src
/// and dst point to the same memory for forward pass, and diff_dst and diff_src
/// point to the same memory for backward pass.
///
/// Batch normalization supports different flavors controlled by
/// mkldnn_batch_normalization_desc_t.  For example, batch normalization can
/// compute the mean and variance on its own or take them as inputs.  It can
/// either perform scaling and shifting using gamma and beta parameters or not.
/// Optionally, it can also perform a fused ReLU, which in case of training
/// would also require a workspace.
///
/// @sa @ref dev_guide_batch_normalization in developer guide
/// @sa @ref c_api_batch_normalization in @ref c_api
/// @{

/// Batch normalization for forward propagation.  Implements descriptor,
/// primitive descriptor, and primitive.
struct batch_normalization_forward : public primitive {

    /// Descriptor for batch normalization forward propagation.
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        template <typename T>

        /// Initializes a batch normalization descriptor for forward propagation
        /// using @p prop_kind (possible values are #mkldnn::forward_training and
        /// #mkldnn::forward_inference), memory descriptor @p data_desc,
        /// normalization parameter @p epsilon, and @p flags set using bit flags
        /// of type mkldnn_batch_normalization_desc_t.
        ///
        /// @note In-place operation is supported; that is, dst points to the
        ///       same memory as src.
        desc(prop_kind aprop_kind, const memory::desc &src_desc, T epsilon,
                batch_normalization_flags flags) {
            error::wrap_c_api(
                    mkldnn_batch_normalization_forward_desc_init(&data,
                            mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                            static_cast<float>(epsilon), convert_to_c(flags)),
                    "could not create a batch normalization forward "
                    "descriptor");
        }
    };

    /// Primitive descriptor for batch normalization forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {

        /// Initializes a primitive descriptor for batch normalization forward
        /// propagation.
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        /// Initializes a primitive descriptor for batch normalization forward
        /// propagation with attributes defined by @p attr.
        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);

        memory::desc mean_desc() const { return stat_desc(mean); }
        memory::desc variance_desc() const { return stat_desc(var); }

    private:
        enum { mean = 1, var = 2, };
        memory::desc stat_desc(int kind) const {
            mkldnn_batch_normalization_desc_t *p;
            error::wrap_c_api(mkldnn_primitive_desc_query(
                    get(), mkldnn::convert_to_c(query::batch_normalization_d), 0, &p),
                    "could not get a batch-normalization descriptor");
            return query_md(p->flags & mkldnn_use_global_stats ? query::src_md
                                                               : query::dst_md,
                    kind);
        }
    };

    batch_normalization_forward(const primitive_desc &pd): primitive(pd) {}
};

/// Batch normalization backward propagation.  Implements descriptor, primitive
/// descriptor, and primitive.
struct batch_normalization_backward : public primitive {

    /// Descriptor for batch normalization backward propagation.
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        template <typename T>

        /// Initializes a batch normalization descriptor for backward
        /// propagation with respect to data and scale-shift parameters using
        /// memory descriptors @p data_desc and @p diff_data_desc, normalization
        /// parameter @p epsilon, and @p flags set using bit flags of type
        /// mkldnn_batch_normalization_desc_t.
        ///
        /// @note In-place operation is supported; that is, diff_src points to
        ///       the same memory as diff_dst.
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, T epsilon,
                batch_normalization_flags flags) {
            error::wrap_c_api(
                    mkldnn_batch_normalization_backward_desc_init(&data,
                            mkldnn::convert_to_c(aprop_kind),
                            &diff_data_desc.data, &data_desc.data,
                            static_cast<float>(epsilon), convert_to_c(flags)),
                    "could not create a batch normalization backward "
                    "descriptor");
        }
    };

    /// Primitive descriptor for batch normalization backward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {

        /// Initializes a primitive descriptor for batch normalization backward
        /// propagation.
        primitive_desc(const desc &desc, const engine &e,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        /// Initializes a primitive descriptor for batch normalization backward
        /// propagation with attributes defined by @p attr.
        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(mean, src, 1);
        REG_QUERY_MD(variance, src, 2);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_weights, diff_weights, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    batch_normalization_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_inner_product Inner Product
/// A primitive to compute an inner product.
///
/// @sa @ref dev_guide_inner_product in developer guide
/// @sa @ref c_api_inner_product in @ref c_api
/// @{

/// Inner product for forward propagation.  Implements descriptor, primitive
/// descriptor, and primitive.
struct inner_product_forward: public primitive {

    /// Initializes an inner product descriptor for forward propagation using
    /// @p prop_kind (possible values are #mkldnn::forward_training and
    /// #mkldnn::forward_inference) and memory descriptors. In order to create
    /// an inner product without bias, @p bias_desc should refer to a
    /// descriptor with memory format kind set to
    /// #mkldnn::memory::format_tag::undef.
    ///
    /// @note Memory descriptors are allowed to be initialized with
    ///       #mkldnn::memory::format_tag::any value of @p format_kind.
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, &bias_desc.data, &dst_desc.data),
                    "could not create a inner product forward descriptor");
        }

        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, nullptr, &dst_desc.data),
                    "could not create a inner product forward descriptor");
        }
    };

    /// Primitive descriptor for inner product forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(bias, weights, 1);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    inner_product_forward(const primitive_desc &pd): primitive(pd) {}
};

/// Inner product for backward propagation with respect to data.  Implements
/// descriptor, primitive descriptor, and primitive.
struct inner_product_backward_data: public primitive {

    /// Initializes an inner product descriptor for backward propagation with
    /// respect to data using memory descriptors.
    ///
    /// @note Memory descriptors are allowed to be initialized with
    ///       #mkldnn::memory::format_tag::any value of @p format_kind.
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_data_desc_init(&data,
                        &diff_src_desc.data, &weights_desc.data,
                        &diff_dst_desc.data),
                "could not create a inner product backward data descriptor");
        }
    };

    /// Primitive descriptor for inner product backward propagation with
    /// respect to data.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const inner_product_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const inner_product_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    inner_product_backward_data(const primitive_desc &pd): primitive(pd) {}
};

/// Inner product for backward propagation with respect to weights.  Implements
/// descriptor, primitive descriptor, and primitive.
struct inner_product_backward_weights: public primitive {

    /// Initializes an inner product descriptor for backward propagation with
    /// respect to weights using memory descriptors.
    ///
    /// @note Memory descriptors are allowed to be initialized with
    ///       #mkldnn::memory::format_tag::any value of @p format_kind.
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_weights_desc_init(
                        &data, &src_desc.data, &diff_weights_desc.data,
                        &diff_bias_desc.data, &diff_dst_desc.data),
                "could not create a inner product backward weights descriptor");
        }
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_weights_desc_init(
                        &data, &src_desc.data, &diff_weights_desc.data,
                        nullptr, &diff_dst_desc.data),
                "could not create a inner product backward weights descriptor");
        }
    };

    /// Primitive descriptor for inner product backward propagation with
    /// respect to weights.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const inner_product_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const inner_product_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(diff_weights, diff_weights, 0);
        REG_QUERY_MD(diff_bias, diff_weights, 1);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    inner_product_backward_weights(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_rnn RNN
/// A primitive to compute common recurrent layer.
///
/// @sa @ref dev_guide_rnn in developer guide
/// @sa @ref c_api_rnn in @ref c_api
/// @{

/// Implements a recurrent cell.
struct rnn_cell {

    /// Descriptor for recurrent cell.
    struct desc {
        mkldnn_rnn_cell_desc_t c_rnn_cell_;

        /// Initializes a recurrent cell descriptor using @p rnn_cell_desc, @p
        /// kind (possible values are #mkldnn::algorithn::vanilla_rnn,
        /// #mkldnn::algorithm::vanilla_lstm, #mkldnn::algorithm::vanilla_gru,
        /// and #mkldnn::algorithm::gru_linear_before_reset), @p activation_f
        /// (possible values are #mkldnn::algorithm::eltwise_relu and
        /// #mkldnn::algorithm::eltwise_tanh).
        desc(algorithm kind, algorithm activation_f) {
            error::wrap_c_api(mkldnn_rnn_cell_desc_init(&c_rnn_cell_,
                        mkldnn::convert_to_c(kind),
                        mkldnn::convert_to_c(activation_f), 0U, 0, 0),
                    "could not init an rnn cell descriptor");
        }
        desc(algorithm kind): desc(kind, algorithm::undef) {}

        operator const mkldnn_rnn_cell_desc_t*() const { return &c_rnn_cell_; }

        algorithm get_cell_kind() const
        { return algorithm(c_rnn_cell_.cell_kind); }
        algorithm get_activation() const
        { return algorithm(c_rnn_cell_.activation_kind); }

        float get_alpha() const { return c_rnn_cell_.alpha; }
        void set_alpha(float alpha) {
            c_rnn_cell_.flags |= mkldnn_rnn_cell_with_relu;
            c_rnn_cell_.alpha = alpha;
        }

        float get_clipping() const { return c_rnn_cell_.clipping; }
        void set_clipping(float clipping) {
            c_rnn_cell_.flags |= mkldnn_rnn_cell_with_clipping;
            c_rnn_cell_.clipping = clipping;
        }

        /// Returns the number of states of a particular RNN cell descriptor.
        int get_gates_count() const {
            return mkldnn_rnn_cell_get_gates_count(&c_rnn_cell_);
        }

        /// Returns the number of states of a particular RNN cell descriptor.
        int get_state_count() const {
            return mkldnn_rnn_cell_get_states_count(&c_rnn_cell_);
        }
    };
};

/// RNN for forward propagation.  Implements descriptor, primitive descriptor,
/// and primitive.
struct rnn_forward : public primitive {

    /// Descriptor for RNN forward propagation.
    struct desc {
        mkldnn_rnn_desc_t data;

        /// Initializes an RNN descriptor for forward propagation using @p
        /// prop_kind, @p rnn_cell_desc, @p direction, and memory descriptors.
        /// @note If @p prop_kind equals #mkldnn::forward_training, you must
        /// query a workspace memory descriptor before creating the primitive.
        ///
        /// @p src_iter_desc, @p bias_desc, and @p dst_iter_desc are allowed
        /// to point to a zero memory descriptor, which would indicate that
        /// the RNN primitive should not use them.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #mkldnn::memory::format_tag::any value of @p
        ///     format_kind.
        desc(prop_kind aprop_kind, rnn_cell::desc cell,
                const rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc
            ) {
            error::wrap_c_api(mkldnn_rnn_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), cell,
                        mkldnn::convert_to_c(direction),
                        &src_layer_desc.data, &src_iter_desc.data,
                        &weights_layer_desc.data, &weights_iter_desc.data,
                        &bias_desc.data,
                        &dst_layer_desc.data, &dst_iter_desc.data),
                    "could not create an RNN forward descriptor");
        }

    };

    /// Primitive descriptor for RNN forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src_layer, src, 0);
        REG_QUERY_MD(src_iter, src, 1);
        REG_QUERY_MD(weights_layer, weights, 0);
        REG_QUERY_MD(weights_iter, weights, 1);
        REG_QUERY_MD(bias, weights, 2);
        REG_QUERY_MD(dst_layer, dst, 0);
        REG_QUERY_MD(dst_iter, dst, 1);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    rnn_forward(const primitive_desc &pd): primitive(pd) {}
};

/// RNN for backward propagation.  Implements descriptor, primitive descriptor,
/// and primitive.
struct rnn_backward : public primitive {

    /// RNN descriptor for backward propagation.
    struct desc {
        mkldnn_rnn_desc_t data;

        /// Initializes an RNN descriptor for backward propagation using @p
        /// prop_kind, @p rnn_cell_desc, @p direction, and memory descriptors.
        ///
        /// @note All memory descriptors are allowed to be initialized with
        ///       #mkldnn::memory::format_tag::any value of @p format_kind.
        ///
        /// @p src_iter_desc (simultaneously with @p diff_src_iter_desc), @p
        /// bias_desc (simultaneously with @p diff_bias_desc), and @p
        /// dst_iter_desc (simultaneously with @p diff_src_iter_desc) are
        /// allowed point to a zero memory descriptor, which would indicate
        /// that the RNN primitive should not use them.
        desc(prop_kind aprop_kind, rnn_cell::desc cell,
                const rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc) {
            error::wrap_c_api(mkldnn_rnn_backward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), cell,
                        mkldnn::convert_to_c(direction),
                        &src_layer_desc.data, &src_iter_desc.data,
                        &weights_layer_desc.data, &weights_iter_desc.data,
                        &bias_desc.data,
                        &dst_layer_desc.data, &dst_iter_desc.data,
                        &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                        &diff_weights_layer_desc.data,
                        &diff_weights_iter_desc.data, &diff_bias_desc.data,
                        &diff_dst_layer_desc.data, &diff_dst_iter_desc.data),
                    "could not create an RNN backward descriptor");
        }

    };

    /// Primitive descriptor for RNN backward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const rnn_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const rnn_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src_layer, src, 0);
        REG_QUERY_MD(src_iter, src, 1);
        REG_QUERY_MD(weights_layer, weights, 0);
        REG_QUERY_MD(weights_iter, weights, 1);
        REG_QUERY_MD(bias, weights, 2);
        REG_QUERY_MD(dst_layer, dst, 0);
        REG_QUERY_MD(dst_iter, dst, 1);
        REG_QUERY_MD(workspace, workspace, 0);

        REG_QUERY_MD(diff_src_layer, diff_src, 0);
        REG_QUERY_MD(diff_src_iter, diff_src, 1);
        REG_QUERY_MD(diff_weights_layer, diff_weights, 0);
        REG_QUERY_MD(diff_weights_iter, diff_weights, 1);
        REG_QUERY_MD(diff_bias, diff_weights, 2);
        REG_QUERY_MD(diff_dst_layer, diff_dst, 0);
        REG_QUERY_MD(diff_dst_iter, diff_dst, 1);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    // With last iteration (with and without input src_iter)
    rnn_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_shuffle Shuffle
/// A primitive to shuffle data along the axis.
///
/// @sa @ref dev_guide_shuffle in developer guide
/// @sa @ref c_api_shuffle in @ref c_api
/// @{

/// Shuffle for forward propagation.  Implements descriptor, primitive
/// descriptor, and primitive.
struct shuffle_forward : public primitive {

    /// Descriptor for shuffle forward propagation.
    struct desc {
        mkldnn_shuffle_desc_t data;

        /// Initializes a shuffle descriptor for forward propagation using @p
        /// prop_kind, memory descriptor @p data_desc, @p axis, and @p
        /// group_size.
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
                int axis, int group_size) {
            error::wrap_c_api(mkldnn_shuffle_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &data_desc.data,
                        axis, group_size),
                    "could not create a shuffle forward descriptor");
        }
    };

    /// Primitive descriptor for shuffle forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const primitive_attr &aattr = primitive_attr())
            : mkldnn::primitive_desc(&desc.data, &aattr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    shuffle_forward(const primitive_desc &pd): primitive(pd) {}
};

/// Shuffle for backward propagation.  Implements descriptor, primitive
/// descriptor, and primitive.
struct shuffle_backward : public primitive {

    // Descriptor for shuffle backward propagation.
    struct desc {
        mkldnn_shuffle_desc_t data;

        /// Initializes a shuffle descriptor for backward propagation using
        /// memory descriptor @p diff_data_desc, @p axis, and @p group_size.
        desc(const memory::desc &diff_data_desc, int axis, int group_size) {
            error::wrap_c_api(mkldnn_shuffle_backward_desc_init(&data,
                        &diff_data_desc.data, axis, group_size),
                    "could not create a shuffle backward descriptor");
        }
    };

    // Primitive descriptor for shuffle backward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const shuffle_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &aattr = primitive_attr())
            : mkldnn::primitive_desc(
                      &desc.data, &aattr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    shuffle_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @} Primitives

/// @} C++ API

#undef REG_QUERY_MD

// implementation section
#ifndef DOXYGEN_SHOULD_SKIP_THIS

inline primitive::primitive(const_mkldnn_primitive_desc_t c_pd) {
    mkldnn_primitive_t result;
    error::wrap_c_api(mkldnn_primitive_create(&result, c_pd),
            "could not create a primitive");
    reset(result);
}

inline primitive::primitive(const primitive_desc &pd): primitive(pd.get()) {}

inline void primitive::execute(stream &astream,
        const std::unordered_map<int, memory> &args) const {
    std::vector<mkldnn_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a: args)
        c_args.push_back({a.first, a.second.get()});

    error::wrap_c_api(mkldnn_primitive_execute(get(), astream.get(),
                (int)c_args.size(), c_args.data()),
            "primitive execution fail");
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace mkldnn

#endif
