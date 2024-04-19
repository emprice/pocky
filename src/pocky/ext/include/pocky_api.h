#ifndef POCKY_API_H
#define POCKY_API_H

typedef struct
{
    PyTypeObject *context_type;
    PyTypeObject *bufpair_type;

    cl_int (*opencl_kernels_from_fragments)(cl_uint, const char **,
        cl_context, cl_program *, cl_uint *, cl_kernel **);
    cl_int (*opencl_kernels_from_files)(cl_uint, const char **,
        cl_context, cl_program *, cl_uint *, cl_kernel **);
    cl_int (*opencl_kernel_lookup_by_name)(cl_uint,
        cl_kernel *, const char *, cl_kernel *);
    cl_int (*opencl_kernels_free_all)(cl_uint *, cl_kernel **);
    cl_int (*opencl_program_free)(cl_program *);

    int (*bufpair_empty_like)(pocky_context_object *,
        pocky_bufpair_object *, pocky_bufpair_object **);
    int (*bufpair_empty_from_shape)(pocky_context_object *,
        size_t, long *, pocky_bufpair_object **);

    const char *(*opencl_error_to_string)(cl_int);
}
pocky_api_object;

#ifndef NO_IMPORT_POCKY

pocky_api_object *pocky_api;

static int import_pocky(void)
{
    pocky_api = (pocky_api_object *) PyCapsule_Import("pocky.ext._C_API", 0);
    return (pocky_api == NULL) ? -1 : 0;
}

#else

extern pocky_api_object *pocky_api;

#endif      /* NO_IMPORT_POCKY */

#endif      /* POCKY_API_H */

/* vim: set ft=c.doxygen: */
