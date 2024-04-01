#include "pocky.h"
#include "utils.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pocky_ARRAY_API
#include <numpy/arrayobject.h>

PyObject *list_all_platforms(PyObject *self, PyObject *Py_UNUSED(args))
{
    cl_int err;
    char buf[BUFSIZ];

    PyObject *result;
    opencl_platform_query_entry *entries, *entry;

    err = opencl_platform_query_all(&entries);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, ocl_fmt_internal, opencl_error_to_string(err), err);
        PyErr_SetString(ocl_error, buf);
        return NULL;
    }

    result = PyList_New(0);
    if (!result) return NULL;

    entry = entries;
    while (entry)
    {
        PyObject *seq, *id, *name, *version;

        seq = PyStructSequence_New(platform_type);
        if (!seq) return NULL;

        id      = PyCapsule_New(entry->id, "PlatformID", NULL);
        name    = PyUnicode_FromString(entry->name);
        version = PyUnicode_FromString(entry->version);

        if ((!id) || (!name) || (!version))
        {
            Py_XDECREF(seq);

            Py_XDECREF(id);
            Py_XDECREF(name);
            Py_XDECREF(version);

            entry = entry->next;
            continue;
        }

        PyStructSequence_SetItem(seq, 0, id);
        PyStructSequence_SetItem(seq, 1, name);
        PyStructSequence_SetItem(seq, 2, version);

        PyList_Append(result, seq);

        entry = entry->next;
    }

    opencl_platform_query_all_free(&entries);
    return result;
}

PyObject *list_all_devices(PyObject *self, PyObject *args)
{
    cl_int err;
    char buf[BUFSIZ];

    cl_platform_id plat_id;
    PyObject *plat, *cap, *result;
    opencl_device_query_entry *entries, *entry;

    if (!PyArg_ParseTuple(args, "O!", platform_type, &plat)) return NULL;

    cap = PyStructSequence_GetItem(plat, 0);
    if (!PyCapsule_CheckExact(cap))
    {
        PyErr_SetString(PyExc_TypeError, ocl_msg_not_a_capsule);
        return NULL;
    }
    plat_id = PyCapsule_GetPointer(cap, "PlatformID");

    err = opencl_device_query_all(plat_id, &entries);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, ocl_fmt_internal, opencl_error_to_string(err), err);
        PyErr_SetString(ocl_error, buf);
        return NULL;
    }

    result = PyList_New(0);
    if (!result) return NULL;

    entry = entries;
    while (entry)
    {
        PyObject *seq, *id, *name, *type, *t;

        seq = PyStructSequence_New(device_type);
        if (!seq) return NULL;

        id   = PyCapsule_New(entry->id, "DeviceID", NULL);
        name = PyUnicode_FromString(entry->name);
        type = PyList_New(0);

        if ((!seq) || (!id) || (!name) || (!type))
        {
            Py_XDECREF(seq);

            Py_XDECREF(id);
            Py_XDECREF(name);
            Py_XDECREF(type);

            entry = entry->next;
            continue;
        }

        if (entry->type & CL_DEVICE_TYPE_CPU)
        {
            t = PyUnicode_FromString("cpu");
            if (!t) Py_XDECREF(t);
            else PyList_Append(type, t);
        }

        if (entry->type & CL_DEVICE_TYPE_GPU)
        {
            t = PyUnicode_FromString("gpu");
            if (!t) Py_XDECREF(t);
            else PyList_Append(type, t);
        }

        if (entry->type & CL_DEVICE_TYPE_ACCELERATOR)
        {
            t = PyUnicode_FromString("accel");
            if (!t) Py_XDECREF(t);
            else PyList_Append(type, t);
        }

        if (entry->type & CL_DEVICE_TYPE_CUSTOM)
        {
            t = PyUnicode_FromString("custom");
            if (!t) Py_XDECREF(t);
            else PyList_Append(type, t);
        }

        PyStructSequence_SetItem(seq, 0, id);
        PyStructSequence_SetItem(seq, 1, name);
        PyStructSequence_SetItem(seq, 2, type);

        PyList_Append(result, seq);

        entry = entry->next;
    }

    opencl_device_query_all_free(&entries);
    return result;
}
