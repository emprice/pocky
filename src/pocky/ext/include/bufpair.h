#ifndef POCKY_BUFPAIR_H
#define POCKY_BUFPAIR_H

typedef struct
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    PyObject_HEAD
#endif  /* DOXYGEN_SHOULD_SKIP_THIS */
    context_object *context;
    PyObject *host;
    cl_mem device;
    size_t host_size;
    size_t device_size;
}
bufpair_object;

extern PyTypeObject bufpair_type;

extern PyObject *bufpair_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
extern int bufpair_init(bufpair_object *self, PyObject *args, PyObject *kwargs);
extern void bufpair_dealloc(bufpair_object *self);

extern PyObject *bufpair_array(bufpair_object *self, PyObject *noargs);

extern PyObject *bufpair_get_host(bufpair_object *self, void *closure);
extern int bufpair_set_host(bufpair_object *self, PyObject *value, void *closure);

extern PyGetSetDef bufpair_getsetters[];
extern PyMethodDef bufpair_methods[];

extern int bufpair_empty_like(context_object *context,
    bufpair_object *like, bufpair_object **bufpair);
extern int bufpair_empty_from_shape(context_object *context,
    size_t ndim, size_t *shape, bufpair_object **bufpair);

#endif      /* POCKY_BUFPAIR_H */
