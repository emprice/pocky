#include "pocky.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pocky_ARRAY_API
#include <numpy/arrayobject.h>

PyTypeObject pocky_bufpair_type;

PyObject *pocky_bufpair_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    pocky_bufpair_object *self;

    if ((self = (pocky_bufpair_object *) type->tp_alloc(type, 0)))
    {
        self->host = NULL;
        self->context = NULL;
        self->device = NULL;
        self->host_size = 0;
        self->device_size = 0;
    }

    return (PyObject *) self;
}

int pocky_bufpair_init(pocky_bufpair_object *self,
        PyObject *args, PyObject *kwargs)
{
    cl_int err;
    PyObject *host, *tmp;
    pocky_context_object *context;
    char *keys[] = { "context", "host", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", keys,
        &pocky_context_type, &context, &PyArray_Type, &host)) return -1;

    if (!PyArray_Check(host) ||
        !(PyArray_FLAGS((PyArrayObject *) host) & NPY_ARRAY_C_CONTIGUOUS) ||
        (PyArray_TYPE((PyArrayObject *) host) != NPY_FLOAT32))
    {
        PyErr_SetString(PyExc_ValueError, "Host array must be C-contiguous "
            "and of dtype float32.");
        return -1;
    }

    tmp = (PyObject *) self->context;
    Py_INCREF(context);
    self->context = context;
    Py_XDECREF(tmp);

    tmp = self->host;
    Py_INCREF(host);
    self->host = host;
    Py_XDECREF(tmp);

    self->host_size = PyArray_SIZE((PyArrayObject *) self->host);
    self->device_size = self->host_size;

    if (self->device != NULL) clReleaseMemObject(self->device);
    self->device = clCreateBuffer(context->ctx, CL_MEM_READ_WRITE,
        self->device_size * sizeof(cl_float), NULL, &err);
    if (err != CL_SUCCESS) return -1;

    return 0;
}

void pocky_bufpair_dealloc(pocky_bufpair_object *self)
{
    Py_XDECREF(self->context);
    Py_XDECREF(self->host);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *pocky_bufpair_array(pocky_bufpair_object *self,
        PyObject *Py_UNUSED(noargs))
{
    if (!self->host)
    {
        PyErr_SetString(PyExc_AttributeError, "host");
        return NULL;
    }

    return Py_NewRef(self->host);
}

PyObject *pocky_bufpair_get_host(pocky_bufpair_object *self, void *closure)
{
    return Py_NewRef(self->host);
}

int pocky_bufpair_set_host(pocky_bufpair_object *self,
        PyObject *value, void *closure)
{
    size_t new_size;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete host array");
        return -1;
    }

    if ((!PyArray_Check((PyArrayObject *) value)) ||
        (!(PyArray_FLAGS((PyArrayObject *) value) & NPY_ARRAY_C_CONTIGUOUS)) ||
        (PyArray_TYPE((PyArrayObject *) value) != NPY_FLOAT32))
    {
        PyErr_SetString(PyExc_ValueError, "Host array must be a C-contiguous "
            "NumPy array of dtype float32");
        return -1;
    }

    new_size = PyArray_SIZE((PyArrayObject *) value);

    if (new_size > self->device_size)
    {
        PyErr_SetString(PyExc_ValueError, "New host array must not be larger "
            "than the original allocation on the device");
        return -1;
    }

    self->host_size = new_size;

    Py_SETREF(self->host, Py_NewRef(value));
    return 0;
}

PyGetSetDef pocky_bufpair_getsetters[] = {
    { "host", (getter) pocky_bufpair_get_host, (setter) pocky_bufpair_set_host,
      "Exposes the underlying NumPy host array. The size of the array cannot "
      "be increased after the BufferPair is created, but the array can be "
      "replaced by one of the same dtype and equal or smaller size.", NULL },
    { NULL }    /* sentinel */
};

PyMethodDef pocky_bufpair_methods[] = {
    { "__array__", (PyCFunction) pocky_bufpair_array, METH_NOARGS,
      "Return the host NumPy array" },
    { NULL, NULL, 0, NULL }    /* sentinel */
};

int pocky_bufpair_empty_like(pocky_context_object *context,
    pocky_bufpair_object *like, pocky_bufpair_object **bufpair)
{
    cl_int err;
    char buf[BUFSIZ];

    /* Create the buffer memory object */
    *bufpair = (pocky_bufpair_object *)
        pocky_bufpair_new(&pocky_bufpair_type, NULL, NULL);
    if (*bufpair == NULL) return -1;

    Py_INCREF(context);
    (*bufpair)->context = context;

    (*bufpair)->host = (PyObject *)
        PyArray_NewLikeArray((PyArrayObject *) like->host, NPY_CORDER, NULL, 1);
    if ((*bufpair)->host == NULL) return -1;

    (*bufpair)->host_size = like->host_size;
    (*bufpair)->device_size = like->device_size;

    (*bufpair)->device = clCreateBuffer(context->ctx, CL_MEM_READ_WRITE,
        (*bufpair)->device_size * sizeof(cl_float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, pocky_ocl_fmt_internal,
            pocky_opencl_error_to_string(err), err);
        PyErr_SetString(pocky_ocl_error, buf);
        return -1;
    }

    return 0;
}

int pocky_bufpair_empty_from_shape(pocky_context_object *context,
    size_t ndim, size_t *shape, pocky_bufpair_object **bufpair)
{
    cl_int err;
    char buf[BUFSIZ];

    /* Create the buffer memory object */
    *bufpair = (pocky_bufpair_object *)
        pocky_bufpair_new(&pocky_bufpair_type, NULL, NULL);
    if (*bufpair == NULL) return -1;

    Py_INCREF(context);
    (*bufpair)->context = context;

    (*bufpair)->host = (PyObject *)
        PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_FLOAT32),
            ndim, shape, NULL, NULL, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if ((*bufpair)->host == NULL) return -1;

    (*bufpair)->host_size = PyArray_SIZE((PyArrayObject *) (*bufpair)->host);
    (*bufpair)->device_size = (*bufpair)->host_size;

    (*bufpair)->device = clCreateBuffer(context->ctx, CL_MEM_READ_WRITE,
        (*bufpair)->device_size * sizeof(cl_float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, pocky_ocl_fmt_internal,
            pocky_opencl_error_to_string(err), err);
        PyErr_SetString(pocky_ocl_error, buf);
        return -1;
    }

    return 0;
}

/* vim: set ft=c.doxygen: */
