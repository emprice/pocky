#ifndef POCKY_BUFPAIR_H
#define POCKY_BUFPAIR_H

typedef struct
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    PyObject_HEAD
#endif  /* DOXYGEN_SHOULD_SKIP_THIS */
    pocky_context_object *context;
    PyObject *host;
    cl_mem device;
    size_t host_size;
    size_t device_size;
}
pocky_bufpair_object;

extern PyTypeObject pocky_bufpair_type;

extern PyObject *pocky_bufpair_new(PyTypeObject *type,
    PyObject *args, PyObject *kwargs);
extern int pocky_bufpair_init(pocky_bufpair_object *self,
    PyObject *args, PyObject *kwargs);
extern void pocky_bufpair_dealloc(pocky_bufpair_object *self);

extern PyObject *pocky_bufpair_array(pocky_bufpair_object *self, PyObject *noargs);

extern PyObject *pocky_bufpair_get_host(pocky_bufpair_object *self, void *closure);
extern int pocky_bufpair_set_host(pocky_bufpair_object *self,
    PyObject *value, void *closure);

extern PyGetSetDef pocky_bufpair_getsetters[];
extern PyMethodDef pocky_bufpair_methods[];

extern int pocky_bufpair_empty_like(pocky_context_object *context,
    pocky_bufpair_object *like, pocky_bufpair_object **bufpair);
extern int pocky_bufpair_empty_from_shape(pocky_context_object *context,
    size_t ndim, long *shape, pocky_bufpair_object **bufpair);

#endif      /* POCKY_BUFPAIR_H */

/* vim: set ft=c.doxygen: */
