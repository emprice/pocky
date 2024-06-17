#ifndef POCKY_CONTEXT_H
#define POCKY_CONTEXT_H

typedef struct
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    PyObject_HEAD
#endif  /* DOXYGEN_SHOULD_SKIP_THIS */
    cl_context ctx;             /**< OpenCL context handle */
    cl_uint num_queues;         /**< Number of command queues */
    cl_command_queue *queues;   /**< Array of command queues */
}
pocky_context_object;

/** Python type object for the @c pocky.ext.Context object type */
extern PyTypeObject pocky_context_type;

extern PyMethodDef pocky_context_methods[];
extern PyGetSetDef pocky_context_getsetters[];

/**
 * @brief Allocates and initializes an empty Python @c pocky.ext.Context object
 * @param[in] type Type of object to allocate
 * @param[in] args Python arguments to be parsed
 * @param[in] kwargs Python keyword arguments to be parsed
 * @return A new Python @c pocky.ext.Context object
 */
extern PyObject *pocky_context_new(PyTypeObject *type,
        PyObject *args, PyObject *kwargs);

/**
 * @brief Deallocates a Python @c pocky.ext.Context object
 * @param[in] self Object to be deallocated
 */
extern void pocky_context_dealloc(pocky_context_object *self);

/**
 * @brief Classmethod to create a context for the default platform
 * and devices, as well as all their command queues
 * @param[in] self Class reference
 * @param[in] args Python arguments to be parsed; any non-empty object
 * is a fatal error
 * @return Python @c pocky.ext.Context object
 */
extern PyObject *pocky_context_default(PyObject *self, PyObject *args);

/**
 * @brief Classmethod to create a context for a list of devices, as
 * well as all their command queues
 * @param[in] self Class reference
 * @param[in] args Python arguments to be parsed; expects exactly one
 * argument that must be a Python @c PyList object containing only
 * Python @c pocky.ext.Device structs
 * @return Python @c pocky.ext.Context object
 */
extern PyObject *pocky_context_from_device_list(PyObject *self, PyObject *args);

#endif      /* POCKY_CONTEXT_H */

/* vim: set ft=c.doxygen: */
