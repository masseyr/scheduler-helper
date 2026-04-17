/*
 * _gap_ext.c — fast gap analysis for coverage matrices
 *
 * Provides three functions operating on a 2-D uint8 zero-mask
 * (shape T×N, C-contiguous, 1 where the original matrix is 0):
 *
 *   max_gap(mask)            → int64 ndarray shape (N,)
 *   gap_count(mask)          → int64 ndarray shape (N,)
 *   gap_count_and_max(mask)  → (int64 ndarray, int64 ndarray) each shape (N,)
 *
 * Build:
 *   python setup_ext.py build_ext --inplace
 *
 * The resulting shared library (_gap_ext.pyd / _gap_ext.so) is imported
 * automatically by tasking_helper.coverage when present.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

/* NumPy C API */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


/* ── helpers ──────────────────────────────────────────────────────────────── */

/*
 * validate_mask(obj, &out_arr, &T, &N)
 *
 * Borrow a reference to a C-contiguous 2-D uint8 array from *obj*.
 * Returns 0 on success, -1 on failure (exception already set).
 */
static int
validate_mask(PyObject *obj, PyArrayObject **arr, npy_intp *T, npy_intp *N)
{
    *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);   /* C-contiguous, aligned */
    if (*arr == NULL)
        return -1;
    if (PyArray_NDIM(*arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "mask must be 2-D");
        Py_DECREF(*arr);
        return -1;
    }
    *T = PyArray_DIM(*arr, 0);
    *N = PyArray_DIM(*arr, 1);
    return 0;
}


/* ── core scan ────────────────────────────────────────────────────────────── */

/*
 * scan_column(col, T, count_out, max_out)
 *
 * Single pass over one column of the zero-mask.  Updates *count_out and
 * *max_out in-place.  col must point to the first element of the column;
 * elements are assumed to be contiguous (stride = 1 byte, i.e. C order).
 */
static inline void
scan_column(const uint8_t *col, npy_intp T, npy_intp stride,
            int64_t *count_out, int64_t *max_out)
{
    int64_t count   = 0;
    int64_t max_len = 0;
    int64_t run     = 0;

    for (npy_intp t = 0; t < T; t++) {
        if (col[t * stride]) {
            run++;
        } else {
            if (run > 0) {
                count++;
                if (run > max_len)
                    max_len = run;
                run = 0;
            }
        }
    }
    /* close final run */
    if (run > 0) {
        count++;
        if (run > max_len)
            max_len = run;
    }

    *count_out = count;
    *max_out   = max_len;
}


/* ── Python-callable functions ────────────────────────────────────────────── */

static PyObject *
py_max_gap(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject      *obj;
    PyArrayObject *arr;
    npy_intp       T, N;

    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    if (validate_mask(obj, &arr, &T, &N) < 0)
        return NULL;

    npy_intp dims[1] = { N };
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    if (out == NULL) { Py_DECREF(arr); return NULL; }

    const uint8_t *data   = (const uint8_t *)PyArray_DATA(arr);
    int64_t       *result = (int64_t *)PyArray_DATA(out);

    for (npy_intp c = 0; c < N; c++) {
        int64_t cnt = 0, mx = 0;
        scan_column(data + c, T, N, &cnt, &mx);
        result[c] = mx;
    }

    Py_DECREF(arr);
    return (PyObject *)out;
}


static PyObject *
py_gap_count(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject      *obj;
    PyArrayObject *arr;
    npy_intp       T, N;

    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    if (validate_mask(obj, &arr, &T, &N) < 0)
        return NULL;

    npy_intp dims[1] = { N };
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    if (out == NULL) { Py_DECREF(arr); return NULL; }

    const uint8_t *data   = (const uint8_t *)PyArray_DATA(arr);
    int64_t       *result = (int64_t *)PyArray_DATA(out);

    for (npy_intp c = 0; c < N; c++) {
        int64_t cnt = 0, mx = 0;
        scan_column(data + c, T, N, &cnt, &mx);
        result[c] = cnt;
    }

    Py_DECREF(arr);
    return (PyObject *)out;
}


static PyObject *
py_gap_count_and_max(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject      *obj;
    PyArrayObject *arr;
    npy_intp       T, N;

    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    if (validate_mask(obj, &arr, &T, &N) < 0)
        return NULL;

    npy_intp dims[1] = { N };
    PyArrayObject *out_cnt = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    PyArrayObject *out_max = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    if (!out_cnt || !out_max) {
        Py_XDECREF(out_cnt);
        Py_XDECREF(out_max);
        Py_DECREF(arr);
        return NULL;
    }

    const uint8_t *data = (const uint8_t *)PyArray_DATA(arr);
    int64_t *r_cnt      = (int64_t *)PyArray_DATA(out_cnt);
    int64_t *r_max      = (int64_t *)PyArray_DATA(out_max);

    for (npy_intp c = 0; c < N; c++) {
        int64_t cnt = 0, mx = 0;
        scan_column(data + c, T, N, &cnt, &mx);
        r_cnt[c] = cnt;
        r_max[c] = mx;
    }

    Py_DECREF(arr);
    return Py_BuildValue("(NN)", out_cnt, out_max);
}


/* ── module definition ────────────────────────────────────────────────────── */

static PyMethodDef GapExtMethods[] = {
    {
        "max_gap",
        py_max_gap,
        METH_VARARGS,
        "max_gap(mask) -> int64 ndarray, shape (N,)\n\n"
        "Maximum gap length per column of a uint8 zero-mask (shape T x N)."
    },
    {
        "gap_count",
        py_gap_count,
        METH_VARARGS,
        "gap_count(mask) -> int64 ndarray, shape (N,)\n\n"
        "Number of gaps per column of a uint8 zero-mask (shape T x N)."
    },
    {
        "gap_count_and_max",
        py_gap_count_and_max,
        METH_VARARGS,
        "gap_count_and_max(mask) -> (counts, maxima), each int64 ndarray shape (N,)\n\n"
        "Return gap count and max gap in a single pass."
    },
    { NULL, NULL, 0, NULL }  /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_gap_ext",            /* module name */
    "Fast gap analysis C extension for tasking_helper.coverage",
    -1,
    GapExtMethods
};

PyMODINIT_FUNC
PyInit__gap_ext(void)
{
    import_array();   /* initialise NumPy C API */
    return PyModule_Create(&moduledef);
}
