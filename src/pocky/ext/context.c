#include "pocky.h"
#include "utils.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pocky_ARRAY_API
#include <numpy/arrayobject.h>

PyTypeObject context_type;

PyObject *context_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    context_object *self;

    if ((self = (context_object *) type->tp_alloc(type, 0)))
    {
        self->ctx = NULL;
        self->queues = NULL;
        self->num_queues = 0;
    }

    return (PyObject *) self;
}

static void context_init(context_object *context)
{
    cl_int err;
    char buf[BUFSIZ];

    err = opencl_queues_create_all(context->ctx,
        &(context->num_queues), &(context->queues));
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ,
            ocl_fmt_internal, opencl_error_to_string(err), err);
        PyErr_SetString(ocl_error, buf);
        return;
    }
}

void context_dealloc(context_object *self)
{
    size_t idx;

    /* release all queue handles */
    for (idx = 0; idx < self->num_queues; ++idx)
        clReleaseCommandQueue(self->queues[idx]);
    free(self->queues);

    /* release other handles */
    clReleaseProgram(self->program);
    clReleaseContext(self->ctx);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *context_default(PyObject *self, PyObject *Py_UNUSED(args))
{
    cl_int err;
    char buf[BUFSIZ];
    context_object *context;

    /* create the object */
    context = (context_object *) context_new(&context_type, NULL, NULL);

    /* get the default context handle */
    err = opencl_context_default(&(context->ctx));
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, ocl_fmt_internal,
            opencl_error_to_string(err), err);
        PyErr_SetString(ocl_error, buf);
        return NULL;
    }

    /* bundle and return */
    context_init(context);
    return (PyObject *) context;
}

static int get_device_ids_from_list(PyObject *devs,
    Py_ssize_t *num_dev_ids, cl_device_id **dev_ids)
{
    Py_ssize_t idx;

    *num_dev_ids = PyList_Size(devs);
    *dev_ids = malloc((*num_dev_ids) * sizeof(cl_device_id));

    for (idx = 0; idx < *num_dev_ids; ++idx)
    {
        PyObject *dev, *cap;

        dev = PyList_GetItem(devs, idx);
        if (!PyObject_TypeCheck(dev, device_type))
        {
            PyErr_SetString(PyExc_TypeError, ocl_msg_not_a_device);
            return -1;
        }

        cap = PyStructSequence_GetItem(dev, 0);
        if (!PyCapsule_CheckExact(cap))
        {
            PyErr_SetString(PyExc_TypeError, ocl_msg_not_a_capsule);
            return -1;
        }

        (*dev_ids)[idx] = PyCapsule_GetPointer(cap, "DeviceID");
    }

    return 0;
}

PyObject *context_from_device_list(PyObject *self, PyObject *args)
{
    cl_int err;
    char buf[BUFSIZ];
    context_object *context;

    PyObject *devs;
    cl_device_id *dev_ids;
    Py_ssize_t num_dev_ids;

    /* expect one argument, a list */
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &devs)) return NULL;
    /* convert to device ids */
    if (get_device_ids_from_list(devs, &num_dev_ids, &dev_ids)) return NULL;

    /* create the object */
    context = (context_object *) context_new(&context_type, NULL, NULL);

    /* create the context */
    err = opencl_context_from_device_list(num_dev_ids, dev_ids, &(context->ctx));
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, ocl_fmt_internal, opencl_error_to_string(err), err);
        PyErr_SetString(ocl_error, buf);
        return NULL;
    }

    /* bundle and return */
    context_init(context);
    return (PyObject *) context;
}

PyMethodDef context_methods[] = {

    /* classmethods for object creation */
    { "default",
      (PyCFunction) context_default,
      METH_NOARGS | METH_CLASS,
      "default() -> Context\n"
      "Create a context with the default OpenCL platform and devices.\n\n"
      "Returns:\n"
      "  An initialized Context\n" },
    { "from_device_list",
      (PyCFunction) context_from_device_list,
      METH_VARARGS | METH_CLASS,
      "from_device_list(devices: list[Device]) -> Context\n"
      "Create a context from a subset of OpenCL devices on the same platform.\n\n"
      "Args:\n"
      "  devices: The devices to include in the returned context\n"
      "Returns:\n"
      "  An initialized Context\n" },

    { NULL, NULL, 0, NULL }    /* sentinel */
};
