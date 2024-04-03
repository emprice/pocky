#ifndef POCKY_UTILS_H
#define POCKY_UTILS_H

/**
 * @brief Linked list for platform handles, useful for querying
 * capabilities and choosing from those options
 */
typedef struct _opencl_platform_query_entry
{
    cl_platform_id id;                          /**< Platform id */
    char *name;                                 /**< Platform name */
    char *version;                              /**< Platform version string */
    struct _opencl_platform_query_entry *next;  /**< Pointer to next item */
}
pocky_opencl_platform_query_entry;

/**
 * @brief Linked list for device handles, useful for querying
 * capabilities and choosing from those options
 */
typedef struct _opencl_device_query_entry
{
    cl_device_id id;                            /**< Device id */
    char *name;                                 /**< Device name */
    cl_device_type type;                        /**< Device type bitfield */
    struct _opencl_device_query_entry *next;    /**< Pointer to next item */
}
pocky_opencl_device_query_entry;

/**
 * @brief Converts the given error code to a nicer string representation
 * @param[in] err Error code as produced internally by OpenCL
 * @return String representation of @c err
 */
extern const char *pocky_opencl_error_to_string(cl_int err);

/**
 * @brief Creates a linked list of all available platforms
 * @param[out] entries Available platforms
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 */
extern cl_int pocky_opencl_platform_query_all(pocky_opencl_platform_query_entry **entries);

/**
 * @brief Frees the resources allocated by @c opencl_platform_query_all
 * @param[in,out] entries Available platforms in linked list form
 *
 * @rst
 * .. important::
 *
 *    On return, :code:`entries` is cleared and invalidated, so it
 *    cannot be dereferenced after this call.
 * @endrst
 */
extern void pocky_opencl_platform_query_all_free(pocky_opencl_platform_query_entry **entries);

/**
 * @brief Creates a linked list of all available devices for a platform
 * @param[in] plat Platform whose devices will be queried
 * @param[out] entries Available devices for @c plat
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 */
extern cl_int pocky_opencl_device_query_all(cl_platform_id plat,
    pocky_opencl_device_query_entry **entries);

/**
 * @brief Frees the resources allocated by @c opencl_device_query_all
 * @param[in,out] entries Available devices in linked list form
 *
 * @rst
 * .. important::
 *
 *    On return, :code:`entries` is cleared and invalidated, so it
 *    cannot be dereferenced after this call.
 * @endrst
 */
extern void pocky_opencl_device_query_all_free(pocky_opencl_device_query_entry **entries);

/**
 * @brief Creates a context with the default platform and devices
 * @param[out] ctx Created context with default configuration
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 *
 * @rst
 * .. important::
 *
 *    The user is responsible for checking the return code before using
 *    the context in :code:`ctx`, since the handle is not valid if the
 *    context could not be created.
 * @endrst
 */
extern cl_int pocky_opencl_context_default(cl_context *ctx);

/**
 * @brief Creates a context with the default platform and its CPU devices
 * @param[out] ctx Created context with default configuration
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 *
 * @rst
 * .. important::
 *
 *    The user is responsible for checking the return code before using
 *    the context in :code:`ctx`, since the handle is not valid if the
 *    context could not be created.
 * @endrst
 */
extern cl_int pocky_opencl_context_default_cpus(cl_context *ctx);

/**
 * @brief Creates a context with the default platform and its GPU devices
 * @param[out] ctx Created context with default configuration
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 *
 * @rst
 * .. important::
 *
 *    The user is responsible for checking the return code before using
 *    the context in :code:`ctx`, since the handle is not valid if the
 *    context could not be created.
 * @endrst
 */
extern cl_int pocky_opencl_context_default_gpus(cl_context *ctx);

/**
 * @brief Creates a context from an array of devices
 * @param[in] num_devices Number of devices in the @c devices array
 * @param[in] devices Array of devices to include in the context
 * @param[out] ctx Created context for these devices
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 *
 * @rst
 * .. important::
 *
 *    The user is responsible for checking the return code before using
 *    the context in :code:`ctx`, since the handle is not valid if the
 *    context for the requested devices could not be created.
 * @endrst
 */
extern cl_int pocky_opencl_context_from_device_list(size_t num_devices,
    cl_device_id *devices, cl_context *ctx);

/**
 * @brief Frees the given context
 * @param[in,out] ctx Context to be freed; it is released and its handle
 * is set to @c NULL on return
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 */
extern cl_int pocky_opencl_context_free(cl_context *ctx);

/**
 * @brief Creates all command queues for the devices in the given context
 * @param[in] ctx Context for the selected devices
 * @param[out] num_queues Number of created command queues
 * @param[out] queues Array of created command queues
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 *
 * @rst
 * .. important::
 *
 *    Memory for the output list :code:`queues` is allocated by this function,
 *    but the user must free that memory to avoid a memory leak.
 * @endrst
 */
extern cl_int pocky_opencl_queues_create_all(cl_context ctx,
    cl_uint *num_queues, cl_command_queue **queues);

/**
 * @brief Frees all command queues in the given array
 * @param[in,out] num_queues Number of command queues in the array,
 * set to zero on return
 * @param[in,out] queues Array of command queues; each is released, the
 * handle is set to @c NULL, and the container itself is freed on return
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 */
extern cl_int pocky_opencl_queues_free_all(cl_uint *num_queues,
        cl_command_queue **queues);

/**
 * @brief Creates and builds all kernels in the given array of fragments for the
 * given context. If errors occur during the build step, the build logs for
 * the failed builds are written to @c stderr.
 * @param[in] nfrags Number of input source fragments in @c frags
 * @param[in] frags Array of fragments to compile and link
 * @param[in] ctx Context for which to build all kernels
 * @param[out] prog Program encapsulating the kernels
 * @param[out] num_kerns Number of compiled and built kernels
 * @param[out] kerns Array of compiled and built kernels
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 *
 * @rst
 * .. important::
 *
 *    The user is responsible for checking the return code before using
 *    the program in :code:`prog` or the kernels in :code:`kerns`, since
 *    any of these handles may not valid if building failed.
 * @endrst
 */
extern cl_int pocky_opencl_kernels_from_fragments(cl_uint nfrags,
    const char **frags, cl_context ctx, cl_program *prog,
    cl_uint *num_kerns, cl_kernel **kerns);

/**
 * @brief Creates and builds all kernels in the given array of files for the
 * given context. If errors occur during the build step, the build logs for
 * the failed builds are written to @c stderr.
 * @param[in] nfiles Number of input source files in @c files
 * @param[in] files Array of filenames to read, compile, and link
 * @param[in] ctx Context for which to build all kernels
 * @param[out] prog Program encapsulating the kernels
 * @param[out] num_kerns Number of compiled and built kernels
 * @param[out] kerns Array of compiled and built kernels
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 *
 * @rst
 * .. important::
 *
 *    The user is responsible for checking the return code before using
 *    the program in :code:`prog` or the kernels in :code:`kerns`, since
 *    any of these handles may not valid if building failed.
 * @endrst
 */
extern cl_int pocky_opencl_kernels_from_files(cl_uint nfiles,
    const char **files, cl_context ctx, cl_program *prog,
    cl_uint *num_kerns, cl_kernel **kerns);

/**
 * @brief Gets the first kernel function from an array with the matching name
 * @param[in] num_kerns Number of kernels in the array
 * @param[in] kerns Array of kernels
 * @param[in] name Name of the desired kernel function
 * @param[out] kern Kernel with the matching name
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 *
 * @rst
 * .. important::
 *
 *    The user is responsible for checking the return code before using
 *    the kernel in :code:`kern`, since the handle is not valid if no
 *    matching kernel was found.
 * @endrst
 */
extern cl_int pocky_opencl_kernel_lookup_by_name(cl_uint num_kerns,
    cl_kernel *kerns, const char *name, cl_kernel *kern);

/**
 * @brief Frees all kernels in the given array
 * @param[in,out] num_kerns Number of kernels in the array,
 * set to zero on return
 * @param[in,out] kerns Array of kernels; each is released, the
 * handle is set to @c NULL, and the container itself is freed on return
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 */
extern cl_int pocky_opencl_kernels_free_all(cl_uint *num_kerns, cl_kernel **kerns);

/**
 * @brief Frees the given program handle
 * @param[in,out] prog Program to free; it is released and the pointer
 * is set to @c NULL on return
 * @return On success, @c CL_SUCCESS, otherwise an OpenCL error code
 */
extern cl_int pocky_opencl_program_free(cl_program *prog);

#endif      /* POCKY_UTILS_H */

/* vim: set ft=c.doxygen: */
