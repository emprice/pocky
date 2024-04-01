#include "pocky.h"

#define PY_ARRAY_UNIQUE_SYMBOL pocky_ARRAY_API
#include <numpy/arrayobject.h>

PyObject *ocl_error;
PyTypeObject *platform_type;
PyTypeObject *device_type;

/** Methods available at the module level */
static PyMethodDef pocky_methods[] = {
    /* OpenCL methods */
    { "list_all_platforms", list_all_platforms, METH_NOARGS,
      "list_all_platforms() -> list[Platform]\n"
      "Get a list of all available OpenCL platforms.\n\n"
      "Returns:\n"
      "  A list of available platforms\n" },
    { "list_all_devices", list_all_devices, METH_VARARGS,
      "list_all_devices(platform: Platform) -> list[Device]\n"
      "Get a list of all available OpenCL devices for a platform.\n\n"
      "Args:\n"
      "  platform: The platform to query for available devices\n\n"
      "Returns:\n"
      "  A list of devices available on the platform\n" },

    { NULL, NULL, 0, NULL }     /* sentinel value */
};

/** Module definition */
struct PyModuleDef pocky_module = {
    PyModuleDef_HEAD_INIT,
    "pocky",                /* module name */
    pocky_module_doc,       /* module documentation */
    -1,                     /* size of per-interpreter state of the module */
    pocky_methods           /* methods table */
};

PyMODINIT_FUNC PyInit_ext(void)
{
    PyObject *mod;

    mod = PyModule_Create(&pocky_module);
    if (!mod) return NULL;

    /* exceptions */
    ocl_error = PyErr_NewException("pocky.OpenCL", NULL, NULL);

    /* structure types */
    platform_type = PyStructSequence_NewType(&platform_desc);
    device_type = PyStructSequence_NewType(&device_desc);

    context_type = (PyTypeObject) {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "pocky.Context",
        .tp_doc = context_type_doc,
        .tp_basicsize = sizeof(context_object),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = context_new,
        .tp_dealloc = (destructor) context_dealloc,
        .tp_methods = context_methods,
    };

    bufpair_type = (PyTypeObject) {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "pocky.BufferPair",
        .tp_doc = bufpair_type_doc,
        .tp_basicsize = sizeof(bufpair_object),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = bufpair_new,
        .tp_init = (initproc) bufpair_init,
        .tp_dealloc = (destructor) bufpair_dealloc,
        .tp_methods = bufpair_methods,
        .tp_getset = bufpair_getsetters,
    };

    if (PyModule_AddObject(mod, "OpenCL", ocl_error) ||
        PyModule_AddType(mod, platform_type) ||
        PyModule_AddType(mod, device_type) ||
        PyType_Ready(&context_type) ||
        PyModule_AddType(mod, &context_type) ||
        PyType_Ready(&bufpair_type) ||
        PyModule_AddType(mod, &bufpair_type))
    {
        Py_XDECREF(ocl_error);
        Py_XDECREF(platform_type);
        Py_XDECREF(device_type);

        Py_DECREF(&context_type);
        Py_DECREF(&bufpair_type);
        Py_DECREF(mod);

        return NULL;
    }

    /* Required to use the NumPy API */
    import_array();

    return mod;
}
