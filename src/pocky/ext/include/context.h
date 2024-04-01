#ifndef POCKY_CONTEXT_H
#define POCKY_CONTEXT_H

typedef struct
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    PyObject_HEAD
#endif  /* DOXYGEN_SHOULD_SKIP_THIS */
    cl_context ctx;             /**< OpenCL context handle */
    cl_program program;         /**< OpenCL program handle */
    cl_uint num_queues;         /**< Number of command queues */
    cl_command_queue *queues;   /**< Array of command queues */
}
context_object;

/** Python type object for the @c Context object type */
extern PyTypeObject context_type;

extern PyMethodDef context_methods[];

/**
 * @brief Allocates and initializes an empty Python @c Context object
 * @param[in] type Type of object to allocate
 * @param[in] args Python arguments to be parsed
 * @param[in] kwargs Python keyword arguments to be parsed
 * @return A new Python @c Context object
 */
extern PyObject *context_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);

/**
 * @brief Deallocates a Python @c Context object
 * @param[in] self Object to be deallocated
 */
extern void context_dealloc(context_object *self);

/**
 * @brief Classmethod to create a context for the default platform
 * and devices, as well as all their command queues
 * @param[in] self Class reference
 * @param[in] args Python arguments to be parsed; any non-empty object
 * is a fatal error
 * @return Python @c Context object
 */
extern PyObject *context_default(PyObject *self, PyObject *args);

/**
 * @brief Classmethod to create a context for a list of devices, as
 * well as all their command queues
 * @param[in] self Class reference
 * @param[in] args Python arguments to be parsed; expects exactly one
 * argument that must be a Python @c PyList object containing only
 * Python @c Device structs
 * @return Python @c Context object
 */
extern PyObject *context_from_devices(PyObject *self, PyObject *args);

#endif      /* POCKY_CONTEXT_H */
