#include "pocky.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pocky_ARRAY_API
#include <numpy/arrayobject.h>

PyTypeObject pocky_context_type;

PyObject *pocky_context_new(PyTypeObject *type,
        PyObject *args, PyObject *kwargs)
{
    pocky_context_object *self;

    if ((self = (pocky_context_object *) type->tp_alloc(type, 0)))
    {
        self->ctx = NULL;
        self->queues = NULL;
        self->num_queues = 0;
    }

    return (PyObject *) self;
}

static void pocky_context_init(pocky_context_object *context)
{
    cl_int err;
    char buf[BUFSIZ];

    err = pocky_opencl_queues_create_all(context->ctx,
        &(context->num_queues), &(context->queues));
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, pocky_ocl_fmt_internal,
            pocky_opencl_error_to_string(err), err);
        PyErr_SetString(pocky_ocl_error, buf);
        return;
    }
}

void pocky_context_dealloc(pocky_context_object *self)
{
    size_t idx;

    /* release all queue handles */
    for (idx = 0; idx < self->num_queues; ++idx)
        clReleaseCommandQueue(self->queues[idx]);
    free(self->queues);

    /* release other handles */
    clReleaseContext(self->ctx);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *pocky_context_default(PyObject *self, PyObject *Py_UNUSED(args))
{
    cl_int err;
    char buf[BUFSIZ];
    pocky_context_object *context;

    /* create the object */
    context = (pocky_context_object *)
        pocky_context_new(&pocky_context_type, NULL, NULL);

    /* get the default context handle */
    err = pocky_opencl_context_default(&(context->ctx));
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, pocky_ocl_fmt_internal,
            pocky_opencl_error_to_string(err), err);
        PyErr_SetString(pocky_ocl_error, buf);
        return NULL;
    }

    /* bundle and return */
    pocky_context_init(context);
    return (PyObject *) context;
}

static int pocky_get_device_ids_from_list(PyObject *devs,
    Py_ssize_t *num_dev_ids, cl_device_id **dev_ids)
{
    Py_ssize_t idx;

    *num_dev_ids = PyList_Size(devs);
    *dev_ids = malloc((*num_dev_ids) * sizeof(cl_device_id));

    for (idx = 0; idx < *num_dev_ids; ++idx)
    {
        PyObject *dev, *cap;

        dev = PyList_GetItem(devs, idx);
        if (!PyObject_TypeCheck(dev, pocky_device_type))
        {
            PyErr_SetString(PyExc_TypeError, pocky_ocl_msg_not_a_device);
            return -1;
        }

        cap = PyStructSequence_GetItem(dev, 0);
        if (!PyCapsule_CheckExact(cap))
        {
            PyErr_SetString(PyExc_TypeError, pocky_ocl_msg_not_a_capsule);
            return -1;
        }

        (*dev_ids)[idx] = PyCapsule_GetPointer(cap, "DeviceID");
    }

    return 0;
}

PyObject *pocky_context_from_device_list(PyObject *self, PyObject *args)
{
    cl_int err;
    char buf[BUFSIZ];
    pocky_context_object *context;

    PyObject *devs;
    cl_device_id *dev_ids;
    Py_ssize_t num_dev_ids;

    /* expect one argument, a list */
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &devs)) return NULL;
    /* convert to device ids */
    if (pocky_get_device_ids_from_list(devs, &num_dev_ids, &dev_ids)) return NULL;

    /* create the object */
    context = (pocky_context_object *)
        pocky_context_new(&pocky_context_type, NULL, NULL);

    /* create the context */
    err = pocky_opencl_context_from_device_list(num_dev_ids, dev_ids, &(context->ctx));
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, pocky_ocl_fmt_internal,
            pocky_opencl_error_to_string(err), err);
        PyErr_SetString(pocky_ocl_error, buf);
        return NULL;
    }

    /* bundle and return */
    pocky_context_init(context);
    return (PyObject *) context;
}

PyObject *pocky_context_get_devices(pocky_context_object *self, void *closure)
{
    cl_int err;
    char buf[BUFSIZ];
    cl_device_id *devices;
    cl_uint num_devices, idx;
    PyObject *result;

    /* query for number of devices in the context */
    err = clGetContextInfo(self->ctx,
        CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, pocky_ocl_fmt_internal,
            pocky_opencl_error_to_string(err), err);
        PyErr_SetString(pocky_ocl_error, buf);
        return NULL;
    }

    /* allocate memory to store device info */
    devices = malloc(num_devices * sizeof(cl_device_id));

    /* query for device info into existing memory */
    err = clGetContextInfo(self->ctx, CL_CONTEXT_DEVICES,
        num_devices * sizeof(cl_device_id), devices, NULL);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, pocky_ocl_fmt_internal,
            pocky_opencl_error_to_string(err), err);
        PyErr_SetString(pocky_ocl_error, buf);
        return NULL;
    }

    result = PyList_New(0);

    for (idx = 0; idx < num_devices; ++idx)
    {
        size_t sz;
        cl_device_type type;
        cl_device_id id = devices[idx];
        PyObject *seq_obj, *id_obj, *name_obj = NULL, *type_obj = NULL, *t_obj;

        id_obj = PyCapsule_New(id, "DeviceID", NULL);

        err = clGetDeviceInfo(id, CL_DEVICE_NAME, 0, NULL, &sz);
        if (err == CL_SUCCESS)
        {
            char *name = malloc(sz * sizeof(char));
            err = clGetDeviceInfo(id, CL_DEVICE_NAME, sz, name, NULL);
            if (err == CL_SUCCESS) name_obj = PyUnicode_FromString(name);
            free(name);
        }

        err = clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
        if (err == CL_SUCCESS)
        {
            type_obj = PyList_New(0);

            if (type & CL_DEVICE_TYPE_CPU)
            {
                t_obj = PyUnicode_FromString("cpu");
                if (!t_obj) Py_XDECREF(t_obj);
                else PyList_Append(type_obj, t_obj);
            }

            if (type & CL_DEVICE_TYPE_GPU)
            {
                t_obj = PyUnicode_FromString("gpu");
                if (!t_obj) Py_XDECREF(t_obj);
                else PyList_Append(type_obj, t_obj);
            }

            if (type & CL_DEVICE_TYPE_ACCELERATOR)
            {
                t_obj = PyUnicode_FromString("accel");
                if (!t_obj) Py_XDECREF(t_obj);
                else PyList_Append(type_obj, t_obj);
            }

            if (type & CL_DEVICE_TYPE_CUSTOM)
            {
                t_obj = PyUnicode_FromString("custom");
                if (!t_obj) Py_XDECREF(t_obj);
                else PyList_Append(type_obj, t_obj);
            }
        }

        seq_obj = PyStructSequence_New(pocky_device_type);

        if ((!seq_obj) || (!id_obj) || (!name_obj) || (!type_obj))
        {
            Py_XDECREF(seq_obj);
            Py_XDECREF(id_obj);
            Py_XDECREF(name_obj);
            Py_XDECREF(type_obj);

            continue;
        }

        PyStructSequence_SetItem(seq_obj, 0, id_obj);
        PyStructSequence_SetItem(seq_obj, 1, name_obj);
        PyStructSequence_SetItem(seq_obj, 2, type_obj);

        PyList_Append(result, seq_obj);
    }

    /* cleanup */
    free(devices);

    return result;
}

PyGetSetDef pocky_context_getsetters[] = {
    { "devices", (getter) pocky_context_get_devices, NULL,
      "This is a stub", NULL },
    { NULL }    /* sentinel */
};

PyMethodDef pocky_context_methods[] = {

    /* classmethods for object creation */
    { "default",
      (PyCFunction) pocky_context_default, METH_NOARGS | METH_CLASS,
      "default() -> Context\n"
      "Create a context with the default OpenCL platform and devices.\n\n"
      "Returns:\n"
      "  An initialized Context\n" },
    { "from_device_list",
      (PyCFunction) pocky_context_from_device_list, METH_VARARGS | METH_CLASS,
      "from_device_list(devices: list[Device]) -> Context\n"
      "Create a context from a subset of OpenCL devices on the same platform.\n\n"
      "Args:\n"
      "  devices: The devices to include in the returned context\n"
      "Returns:\n"
      "  An initialized Context\n" },

    { NULL, NULL, 0, NULL }    /* sentinel */
};

/* vim: set ft=c.doxygen: */
