#include "pocky.h"

#define PY_ARRAY_UNIQUE_SYMBOL pocky_ARRAY_API
#include <numpy/arrayobject.h>

PyObject *pocky_ocl_error;
PyTypeObject *pocky_platform_type;
PyTypeObject *pocky_device_type;

/** Methods available at the module level */
static PyMethodDef pocky_methods[] = {
    /* OpenCL methods */
    { "list_all_platforms", pocky_list_all_platforms, METH_NOARGS,
      "list_all_platforms() -> list[Platform]\n"
      "Get a list of all available OpenCL platforms.\n\n"
      "Returns:\n"
      "  A list of available platforms\n" },
    { "list_all_devices", pocky_list_all_devices, METH_VARARGS,
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
    "pocky.ext",            /* module name */
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
    pocky_ocl_error = PyErr_NewException("pocky.ext.OpenCL", NULL, NULL);

    /* structure types */
    pocky_platform_type = PyStructSequence_NewType(&pocky_platform_desc);
    pocky_device_type = PyStructSequence_NewType(&pocky_device_desc);

    pocky_context_type = (PyTypeObject) {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "pocky.ext.Context",
        .tp_doc = pocky_context_type_doc,
        .tp_basicsize = sizeof(pocky_context_object),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = pocky_context_new,
        .tp_dealloc = (destructor) pocky_context_dealloc,
        .tp_methods = pocky_context_methods,
    };

    pocky_bufpair_type = (PyTypeObject) {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "pocky.ext.BufferPair",
        .tp_doc = pocky_bufpair_type_doc,
        .tp_basicsize = sizeof(pocky_bufpair_object),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = pocky_bufpair_new,
        .tp_init = (initproc) pocky_bufpair_init,
        .tp_dealloc = (destructor) pocky_bufpair_dealloc,
        .tp_methods = pocky_bufpair_methods,
        .tp_getset = pocky_bufpair_getsetters,
    };

    if (PyModule_AddObject(mod, "OpenCL", pocky_ocl_error) ||
        PyModule_AddType(mod, pocky_platform_type) ||
        PyModule_AddType(mod, pocky_device_type) ||
        PyType_Ready(&pocky_context_type) ||
        PyModule_AddType(mod, &pocky_context_type) ||
        PyType_Ready(&pocky_bufpair_type) ||
        PyModule_AddType(mod, &pocky_bufpair_type))
    {
        Py_XDECREF(pocky_ocl_error);
        Py_XDECREF(pocky_platform_type);
        Py_XDECREF(pocky_device_type);

        Py_DECREF(mod);
        return NULL;
    }

    /* Required to use the NumPy API */
    import_array();

    return mod;
}

/* vim: set ft=c.doxygen: */
