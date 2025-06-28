#include <string>
#include <Python.h>
#include "HalideRuntime.h"

struct halide_buffer_t;
struct halide_filter_metadata_t;

#ifndef HALIDE_MUST_USE_RESULT
#ifdef __has_attribute
#if __has_attribute(nodiscard)
#define HALIDE_MUST_USE_RESULT [[nodiscard]]
#elif __has_attribute(warn_unused_result)
#define HALIDE_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define HALIDE_MUST_USE_RESULT
#endif
#else
#define HALIDE_MUST_USE_RESULT
#endif
#endif

#ifndef HALIDE_FUNCTION_ATTRS
#define HALIDE_FUNCTION_ATTRS
#endif



#ifdef __cplusplus
extern "C" {
#endif

HALIDE_FUNCTION_ATTRS
int sep_conv(struct halide_buffer_t *_input_feature_map_buffer, struct halide_buffer_t *_kernel_buffer, struct halide_buffer_t *_pw_filter_buffer, struct halide_buffer_t *_bias_buffer, struct halide_buffer_t *_pw_buffer);

HALIDE_FUNCTION_ATTRS
int sep_conv_argv(void **args);

HALIDE_FUNCTION_ATTRS
const struct halide_filter_metadata_t *sep_conv_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif


namespace Halide::PythonRuntime {
extern bool unpack_buffer(PyObject *py_obj,
                          int py_getbuffer_flags,
                          const char *name,
                          int dimensions,
                          Py_buffer &py_buf,
                          halide_dimension_t *halide_dim,
                          halide_buffer_t &halide_buf,
                          bool &py_buf_valid,
                          bool &needs_device_free);
}  // namespace Halide::PythonRuntime

namespace {

template<int dimensions>
struct PyHalideBuffer {
    // Must allocate at least 1, even if d=0
    static constexpr int dims_to_allocate = (dimensions < 1) ? 1 : dimensions;

    Py_buffer py_buf;
    halide_dimension_t halide_dim[dims_to_allocate];
    halide_buffer_t halide_buf;
    bool py_buf_needs_release = false;
    bool needs_device_free = false;

    bool unpack(PyObject *py_obj, int py_getbuffer_flags, const char *name) {
        return Halide::PythonRuntime::unpack_buffer(py_obj, py_getbuffer_flags,
            name, dimensions, py_buf, halide_dim, halide_buf, py_buf_needs_release,
            needs_device_free);
    }

    ~PyHalideBuffer() {
        if (needs_device_free) {
            halide_device_free(nullptr, &halide_buf);
        }
        if (py_buf_needs_release) {
            PyBuffer_Release(&py_buf);
        }
    }

    PyHalideBuffer() = default;
    PyHalideBuffer(const PyHalideBuffer &other) = delete;
    PyHalideBuffer &operator=(const PyHalideBuffer &other) = delete;
    PyHalideBuffer(PyHalideBuffer &&other) = delete;
    PyHalideBuffer &operator=(PyHalideBuffer &&other) = delete;
};

}  // namespace

namespace Halide::PythonExtensions {

namespace {

const char* const sep_conv_kwlist[] = {
  "input_feature_map",
  "kernel",
  "pw_filter",
  "bias",
  "pw",
  nullptr
};

}  // namespace

// sep_conv
PyObject *sep_conv(PyObject *module, PyObject *args, PyObject *kwargs) {
  PyObject* py_input_feature_map;
  PyObject* py_kernel;
  PyObject* py_pw_filter;
  PyObject* py_bias;
  PyObject* py_pw;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO", (char**)sep_conv_kwlist
    , &py_input_feature_map
    , &py_kernel
    , &py_pw_filter
    , &py_bias
    , &py_pw
  )) {
    PyErr_Format(PyExc_ValueError, "Internal error");
    return nullptr;
  }
  PyHalideBuffer<4> b_input_feature_map;
  PyHalideBuffer<4> b_kernel;
  PyHalideBuffer<3> b_pw_filter;
  PyHalideBuffer<2> b_bias;
  PyHalideBuffer<4> b_pw;
  if (!b_input_feature_map.unpack(py_input_feature_map, 0, sep_conv_kwlist[0])) return nullptr;
  if (!b_kernel.unpack(py_kernel, 0, sep_conv_kwlist[1])) return nullptr;
  if (!b_pw_filter.unpack(py_pw_filter, 0, sep_conv_kwlist[2])) return nullptr;
  if (!b_bias.unpack(py_bias, 0, sep_conv_kwlist[3])) return nullptr;
  if (!b_pw.unpack(py_pw, PyBUF_WRITABLE, sep_conv_kwlist[4])) return nullptr;

  b_input_feature_map.halide_buf.set_host_dirty();
  b_kernel.halide_buf.set_host_dirty();
  b_pw_filter.halide_buf.set_host_dirty();
  b_bias.halide_buf.set_host_dirty();
  int result;
  Py_BEGIN_ALLOW_THREADS
  result = sep_conv(
    &b_input_feature_map.halide_buf,
    &b_kernel.halide_buf,
    &b_pw_filter.halide_buf,
    &b_bias.halide_buf,
    &b_pw.halide_buf
  );
  Py_END_ALLOW_THREADS
  if (result == 0) result = halide_copy_to_host(nullptr, &b_pw.halide_buf);
  if (result != 0) {
    #ifndef HALIDE_PYTHON_EXTENSION_OMIT_ERROR_AND_PRINT_HANDLERS
    PyErr_Format(PyExc_RuntimeError, "Halide Runtime Error: %d", result);
    #else
    PyErr_Format(PyExc_ValueError, "Halide error %d", result);
    #endif  // HALIDE_PYTHON_EXTENSION_OMIT_ERROR_AND_PRINT_HANDLERS
    return nullptr;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

}  // namespace Halide::PythonExtensions

#ifndef HALIDE_PYTHON_EXTENSION_OMIT_MODULE_DEFINITION

#ifndef HALIDE_PYTHON_EXTENSION_MODULE_NAME
#define HALIDE_PYTHON_EXTENSION_MODULE_NAME sep_conv
#endif  // HALIDE_PYTHON_EXTENSION_MODULE_NAME

#ifndef HALIDE_PYTHON_EXTENSION_FUNCTIONS
#define HALIDE_PYTHON_EXTENSION_FUNCTIONS X(sep_conv)
#endif  // HALIDE_PYTHON_EXTENSION_FUNCTIONS


static_assert(PY_MAJOR_VERSION >= 3, "Python bindings for Halide require Python 3+");

namespace Halide::PythonExtensions {
#define X(name) extern PyObject *name(PyObject *module, PyObject *args, PyObject *kwargs);
      HALIDE_PYTHON_EXTENSION_FUNCTIONS
#undef X
}  // namespace Halide::PythonExtensions

namespace {

#define _HALIDE_STRINGIFY(x)            #x
#define _HALIDE_EXPAND_AND_STRINGIFY(x) _HALIDE_STRINGIFY(x)
#define _HALIDE_CONCAT(x, y)            x##y
#define _HALIDE_EXPAND_AND_CONCAT(x, y) _HALIDE_CONCAT(x, y)

PyMethodDef _methods[] = {
  #define X(name) {#name, reinterpret_cast<PyCFunction>(Halide::PythonExtensions::name), METH_VARARGS | METH_KEYWORDS, nullptr},
  HALIDE_PYTHON_EXTENSION_FUNCTIONS
  #undef X
  {0, 0, 0, nullptr},  // sentinel
};

PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,                                              // base
    _HALIDE_EXPAND_AND_STRINGIFY(HALIDE_PYTHON_EXTENSION_MODULE_NAME),  // name
    nullptr,                                                            // doc
    -1,                                                                 // size
    _methods,                                                           // methods
    nullptr,                                                            // slots
    nullptr,                                                            // traverse
    nullptr,                                                            // clear
    nullptr,                                                            // free
};

#ifndef HALIDE_PYTHON_EXTENSION_OMIT_ERROR_AND_PRINT_HANDLERS
void _module_halide_error(void *user_context, const char *msg) {
    // Most Python code probably doesn't want to log the error text to stderr,
    // so we won't do that by default.
    #ifdef HALIDE_PYTHON_EXTENSION_LOG_ERRORS_TO_STDERR
    PyGILState_STATE s = PyGILState_Ensure();
    PySys_FormatStderr("%s\n", msg);
    PyGILState_Release(s);
    #endif
}

void _module_halide_print(void *user_context, const char *msg) {
    PyGILState_STATE s = PyGILState_Ensure();
    PySys_FormatStdout("%s", msg);
    PyGILState_Release(s);
}
#endif  // HALIDE_PYTHON_EXTENSION_OMIT_ERROR_AND_PRINT_HANDLERS

}  // namespace

namespace Halide::PythonRuntime {

bool unpack_buffer(PyObject *py_obj,
                   int py_getbuffer_flags,
                   const char *name,
                   int dimensions,
                   Py_buffer &py_buf,
                   halide_dimension_t *halide_dim,
                   halide_buffer_t &halide_buf,
                   bool &py_buf_valid,
                   bool &needs_device_free) {
    py_buf_valid = false;
    needs_device_free = false;

    memset(&py_buf, 0, sizeof(py_buf));
    if (PyObject_GetBuffer(py_obj, &py_buf, PyBUF_FORMAT | PyBUF_STRIDED_RO | PyBUF_ANY_CONTIGUOUS | py_getbuffer_flags) < 0) {
        PyErr_Format(PyExc_ValueError, "Invalid argument %s: Expected %d dimensions, got %d", name, dimensions, py_buf.ndim);
        return false;
    }
    py_buf_valid = true;

    if (dimensions && py_buf.ndim != dimensions) {
        PyErr_Format(PyExc_ValueError, "Invalid argument %s: Expected %d dimensions, got %d", name, dimensions, py_buf.ndim);
        return false;
    }
    /* We'll get a buffer that's either:
     * C_CONTIGUOUS (last dimension varies the fastest, i.e., has stride=1) or
     * F_CONTIGUOUS (first dimension varies the fastest, i.e., has stride=1).
     * The latter is preferred, since it's already in the format that Halide
     * needs. It can can be achieved in numpy by passing order='F' during array
     * creation. However, if we do get a C_CONTIGUOUS buffer, flip the dimensions
     * (transpose) so we can process it without having to reallocate.
     */
    int i, j, j_step;
    if (PyBuffer_IsContiguous(&py_buf, 'F')) {
        j = 0;
        j_step = 1;
    } else if (PyBuffer_IsContiguous(&py_buf, 'C')) {
        j = py_buf.ndim - 1;
        j_step = -1;
    } else {
        /* Python checks all dimensions and strides, so this typically indicates
         * a bug in the array's buffer protocol. */
        PyErr_Format(PyExc_ValueError, "Invalid buffer: neither C nor Fortran contiguous");
        return false;
    }
    for (i = 0; i < py_buf.ndim; ++i, j += j_step) {
        halide_dim[i].min = 0;
        halide_dim[i].stride = (int)(py_buf.strides[j] / py_buf.itemsize);  // strides is in bytes
        halide_dim[i].extent = (int)py_buf.shape[j];
        halide_dim[i].flags = 0;
        if (py_buf.suboffsets && py_buf.suboffsets[i] >= 0) {
            // Halide doesn't support arrays of pointers. But we should never see this
            // anyway, since we specified PyBUF_STRIDED.
            PyErr_Format(PyExc_ValueError, "Invalid buffer: suboffsets not supported");
            return false;
        }
    }
    if (halide_dim[py_buf.ndim - 1].extent * halide_dim[py_buf.ndim - 1].stride * py_buf.itemsize != py_buf.len) {
        PyErr_Format(PyExc_ValueError, "Invalid buffer: length %ld, but computed length %ld",
                     py_buf.len, py_buf.shape[0] * py_buf.strides[0]);
        return false;
    }

    memset(&halide_buf, 0, sizeof(halide_buf));
    needs_device_free = true;
    if (!py_buf.format) {
        halide_buf.type.code = halide_type_uint;
        halide_buf.type.bits = 8;
    } else {
        /* Convert struct type code. See
         * https://docs.python.org/2/library/struct.html#module-struct */
        char *p = py_buf.format;
        while (strchr("@<>!=", *p)) {
            p++;  // ignore little/bit endian (and alignment)
        }
        if (*p == 'f' || *p == 'd' || *p == 'e') {
            // 'f', 'd', and 'e' are float, double, and half, respectively.
            halide_buf.type.code = halide_type_float;
        } else if (*p >= 'a' && *p <= 'z') {
            // lowercase is signed int.
            halide_buf.type.code = halide_type_int;
        } else {
            // uppercase is unsigned int.
            halide_buf.type.code = halide_type_uint;
        }
        const char *type_codes = "bBhHiIlLqQfde";  // integers and floats
        if (*p == '?') {
            // Special-case bool, so that it is a distinct type vs uint8_t
            // (even though the memory layout is identical)
            halide_buf.type.bits = 1;
        } else if (strchr(type_codes, *p)) {
            halide_buf.type.bits = (uint8_t)py_buf.itemsize * 8;
        } else {
            // We don't handle 's' and 'p' (char[]) and 'P' (void*)
            PyErr_Format(PyExc_ValueError, "Invalid data type for %s: %s", name, py_buf.format);
            return false;
        }
    }
    halide_buf.type.lanes = 1;
    halide_buf.dimensions = py_buf.ndim;
    halide_buf.dim = halide_dim;
    halide_buf.host = (uint8_t *)py_buf.buf;

    return true;
}

}  // namespace Halide::PythonRuntime

extern "C" {

HALIDE_EXPORT_SYMBOL PyObject *_HALIDE_EXPAND_AND_CONCAT(PyInit_, HALIDE_PYTHON_EXTENSION_MODULE_NAME)() {
    PyObject *m = PyModule_Create(&_moduledef);
    #ifndef HALIDE_PYTHON_EXTENSION_OMIT_ERROR_AND_PRINT_HANDLERS
    halide_set_error_handler(_module_halide_error);
    halide_set_custom_print(_module_halide_print);
    #endif  // HALIDE_PYTHON_EXTENSION_OMIT_ERROR_AND_PRINT_HANDLERS
    return m;
}

}  // extern "C"
#endif  // HALIDE_PYTHON_EXTENSION_OMIT_MODULE_DEFINITION
