// quill_fastsort_simd._impl — single-threaded radix sort backend (SIMD-friendly).
// Exposes the exact API quill/_backends.py::SimdCompanionBackend probes:
//   sort_{i64,u64,i32,u32,f64,f32}(ndarray)  — sorts in place, ascending.
// Portable C++17 (macOS / Linux / Windows). Kernel: quillcore::serial_sort.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "quill_core.hpp"

using namespace quillcore;

// Sort a writable, C-contiguous buffer of element type T in place. The Quill
// dispatcher guarantees contiguity, writeability and NaN-free float data before
// calling, so we only defend against a mismatched itemsize (wrong entry point).
template <class T>
static PyObject* run(PyObject* arg) {
    Py_buffer v;
    if (PyObject_GetBuffer(arg, &v, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0)
        return NULL;
    if (v.itemsize != 0 && (size_t)v.itemsize != sizeof(T)) {
        PyBuffer_Release(&v);
        PyErr_Format(PyExc_TypeError,
                     "quill_fastsort_simd: expected %zu-byte elements, got %zd",
                     sizeof(T), (Py_ssize_t)v.itemsize);
        return NULL;
    }
    if (v.len % (Py_ssize_t)sizeof(T) != 0) {
        PyBuffer_Release(&v);
        PyErr_SetString(PyExc_ValueError, "buffer length not a multiple of itemsize");
        return NULL;
    }
    size_t n = (size_t)v.len / sizeof(T);
    T* data = (T*)v.buf;
    Py_BEGIN_ALLOW_THREADS
    serial_sort<T>(data, n);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&v);
    Py_RETURN_NONE;
}

static PyObject* sort_i64(PyObject*, PyObject* a) { return run<int64_t>(a); }
static PyObject* sort_u64(PyObject*, PyObject* a) { return run<uint64_t>(a); }
static PyObject* sort_i32(PyObject*, PyObject* a) { return run<int32_t>(a); }
static PyObject* sort_u32(PyObject*, PyObject* a) { return run<uint32_t>(a); }
static PyObject* sort_f64(PyObject*, PyObject* a) { return run<double>(a); }
static PyObject* sort_f32(PyObject*, PyObject* a) { return run<float>(a); }

static PyMethodDef Methods[] = {
    {"sort_i64", sort_i64, METH_O, "in-place ascending sort, int64"},
    {"sort_u64", sort_u64, METH_O, "in-place ascending sort, uint64"},
    {"sort_i32", sort_i32, METH_O, "in-place ascending sort, int32"},
    {"sort_u32", sort_u32, METH_O, "in-place ascending sort, uint32"},
    {"sort_f64", sort_f64, METH_O, "in-place ascending sort, float64"},
    {"sort_f32", sort_f32, METH_O, "in-place ascending sort, float32"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_impl",
    "Quill single-threaded radix sort backend (portable C++).", -1, Methods,
    NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__impl(void) { return PyModule_Create(&moduledef); }
