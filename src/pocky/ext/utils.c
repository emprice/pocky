#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char opencl_msg_success[]             = "CL_SUCCESS";
static const char opencl_msg_invalid_value[]       = "CL_INVALID_VALUE";
static const char opencl_msg_invalid_platform[]    = "CL_INVALID_PLATFORM";
static const char opencl_msg_invalid_device[]      = "CL_INVALID_DEVICE";
static const char opencl_msg_invalid_device_type[] = "CL_INVALID_DEVICE_TYPE";
static const char opencl_msg_invalid_program[]     = "CL_INVALID_PROGRAM";
static const char opencl_msg_invalid_build_opts[]  = "CL_INVALID_BUILD_OPTIONS";
static const char opencl_msg_compiler_unavail[]    = "CL_COMPILER_NOT_AVAILABLE";
static const char opencl_msg_build_failure[]       = "CL_BUILD_PROGRAM_FAILURE";
static const char opencl_msg_device_not_found[]    = "CL_DEVICE_NOT_FOUND";
static const char opencl_msg_out_of_resources[]    = "CL_OUT_OF_RESOURCES";
static const char opencl_msg_out_of_host_memory[]  = "CL_OUT_OF_HOST_MEMORY";
static const char opencl_msg_unknown[]             = "(not a recognized code)";

const char *opencl_error_to_string(cl_int err)
{
    switch (err)
    {
        case CL_SUCCESS:                return opencl_msg_success;
        case CL_INVALID_VALUE:          return opencl_msg_invalid_value;
        case CL_INVALID_PLATFORM:       return opencl_msg_invalid_platform;
        case CL_INVALID_DEVICE:         return opencl_msg_invalid_device;
        case CL_INVALID_DEVICE_TYPE:    return opencl_msg_invalid_device_type;
        case CL_INVALID_PROGRAM:        return opencl_msg_invalid_program;
        case CL_INVALID_BUILD_OPTIONS:  return opencl_msg_invalid_build_opts;
        case CL_COMPILER_NOT_AVAILABLE: return opencl_msg_compiler_unavail;
        case CL_BUILD_PROGRAM_FAILURE:  return opencl_msg_build_failure;
        case CL_DEVICE_NOT_FOUND:       return opencl_msg_device_not_found;
        case CL_OUT_OF_RESOURCES:       return opencl_msg_out_of_resources;
        case CL_OUT_OF_HOST_MEMORY:     return opencl_msg_out_of_host_memory;
    }

    return opencl_msg_unknown;
}

cl_int opencl_platform_query_all(opencl_platform_query_entry **entries)
{
    cl_int err;
    cl_uint idx;
    cl_uint num_platforms;
    cl_platform_id *platforms;
    opencl_platform_query_entry *prev = NULL;

    /* Query for the number of platforms */
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) return err;

    /* Allocate a sufficiently large array and query for the platform ids */
    platforms = malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS)
    {
        free(platforms);
        return err;
    }

    for (idx = 0; idx < num_platforms; ++idx)
    {
        size_t sz;

        cl_platform_id id = platforms[idx];
        opencl_platform_query_entry *entry;

        entry = malloc(sizeof(opencl_platform_query_entry));

        entry->id = id;
        entry->next = prev;

        /* Query for the platform name (first length, then data) */
        err = clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, NULL, &sz);
        if (err == CL_SUCCESS)
        {
            entry->name = malloc(sz * sizeof(char));
            clGetPlatformInfo(id, CL_PLATFORM_NAME, sz, entry->name, NULL);
        }

        /* Query for the platform version (first length, then data) */
        err = clGetPlatformInfo(id, CL_PLATFORM_VERSION, 0, NULL, &sz);
        if (err == CL_SUCCESS)
        {
            entry->version = malloc(sz * sizeof(char));
            clGetPlatformInfo(id, CL_PLATFORM_VERSION, sz, entry->version, NULL);
        }

        prev = entry;
    }

    *entries = prev;

    /* Clean up allocated memory */
    free(platforms);
    return err;
}

void opencl_platform_query_all_free(opencl_platform_query_entry **entries)
{
    opencl_platform_query_entry *next, *entry = *entries;

    while (entry)
    {
        free(entry->name);
        free(entry->version);

        next = entry->next;
        free(entry);
        entry = next;
    }

    *entries = NULL;
}

cl_int opencl_device_query_all(cl_platform_id plat,
    opencl_device_query_entry **entries)
{
    cl_int err;
    cl_uint idx;
    cl_uint num_devices;
    cl_device_id *devices;
    opencl_device_query_entry *prev = NULL;

    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS) return err;

    devices = malloc(num_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS)
    {
        free(devices);
        return err;
    }

    for (idx = 0; idx < num_devices; ++idx)
    {
        size_t sz;

        cl_device_id id = devices[idx];
        opencl_device_query_entry *entry;

        entry = malloc(sizeof(opencl_device_query_entry));

        entry->id = id;
        entry->next = prev;

        err = clGetDeviceInfo(id, CL_DEVICE_NAME, 0, NULL, &sz);
        if (err == CL_SUCCESS)
        {
            entry->name = malloc(sz * sizeof(char));
            clGetDeviceInfo(id, CL_DEVICE_NAME, sz, entry->name, NULL);
        }

        err = clGetDeviceInfo(id, CL_DEVICE_TYPE, 0, NULL, &sz);
        if (err == CL_SUCCESS)
        {
            clGetDeviceInfo(id, CL_DEVICE_TYPE, sz, &(entry->type), NULL);
        }

        prev = entry;
    }

    *entries = prev;

    free(devices);
    return err;
}

void opencl_device_query_all_free(opencl_device_query_entry **entries)
{
    opencl_device_query_entry *next, *entry = *entries;

    while (entry)
    {
        free(entry->name);

        next = entry->next;
        free(entry);
        entry = next;
    }

    *entries = NULL;
}

cl_int opencl_context_default(cl_context *ctx)
{
    cl_int err;
    cl_device_id *devices;
    cl_platform_id platform;
    cl_uint num_devices, num_platforms;
    cl_context_properties props[3];

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) return err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) return err;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS) return err;

    devices = malloc(num_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS) return err;

    props[0] = CL_CONTEXT_PLATFORM;
    props[1] = (cl_context_properties) platform;
    props[2] = 0;
    *ctx = clCreateContext(props, num_devices, devices, NULL, NULL, &err);

    free(devices);
    return err;
}

cl_int opencl_context_from_device_list(size_t num_devices,
    cl_device_id *devices, cl_context *ctx)
{
    cl_int err;
    *ctx = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
    return err;
}

cl_int opencl_queues_create_all(cl_context ctx,
    cl_uint *num_queues, cl_command_queue **queues)
{
    cl_int err;
    size_t idx;
    cl_device_id *devices;

    err = clGetContextInfo(ctx, CL_CONTEXT_NUM_DEVICES,
        sizeof(cl_uint), num_queues, NULL);
    if (err != CL_SUCCESS) return err;

    devices = malloc((*num_queues) * sizeof(cl_device_id));
    err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES,
        (*num_queues) * sizeof(cl_device_id), devices, NULL);
    if (err != CL_SUCCESS)
    {
        free(devices);
        return err;
    }

    *queues = malloc((*num_queues) * sizeof(cl_command_queue));

    for (idx = 0; idx < *num_queues; ++idx)
    {
        cl_command_queue queue;
        cl_device_id dev = devices[idx];

        queue = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
        if (err != CL_SUCCESS)
        {
            free(devices);
            free(*queues);
            return err;
        }

        (*queues)[idx] = queue;
    }

    free(devices);
    return err;
}

cl_int opencl_queues_free_all(cl_uint *num_queues, cl_command_queue **queues)
{
    cl_uint i;
    cl_int err;
    cl_command_queue *q = *queues;

    for (i = 0; i < *num_queues; ++i)
    {
        err = clReleaseCommandQueue(q[i]);
        if (err != CL_SUCCESS) return err;
        q[i] = NULL;
    }

    *queues = NULL;
    *num_queues = 0;
    return err;
}

cl_int opencl_context_default_cpus(cl_context *ctx)
{
    cl_int err;
    *ctx = clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU, NULL, NULL, &err);
    return err;
}

cl_int opencl_context_default_gpus(cl_context *ctx)
{
    cl_int err;
    *ctx = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
    return err;
}

cl_int opencl_context_free(cl_context *ctx)
{
    cl_int err;
    err = clReleaseContext(*ctx);
    *ctx = NULL;
    return err;
}

cl_int opencl_kernels_from_fragments(cl_uint nfrags,
    const char **frags, cl_context ctx, cl_program *prog,
    cl_uint *num_kerns, cl_kernel **kerns)
{
    size_t *lens;
    cl_int err, err1;

    lens = malloc(nfrags * sizeof(size_t));
    memset(lens, 0, nfrags * sizeof(size_t));

    *prog = clCreateProgramWithSource(ctx, nfrags, frags, lens, &err);
    if (err == CL_SUCCESS)
    {
        const char *opts = "-cl-std=CL3.0 -cl-single-precision-constant "
            "-cl-fast-relaxed-math";
        err = clBuildProgram(*prog, 0, NULL, opts, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            cl_uint i, num_devs;
            cl_device_id *devs;

            err1 = clGetProgramInfo(*prog, CL_PROGRAM_NUM_DEVICES,
                sizeof(cl_uint), &num_devs, NULL);
            if (err1 == CL_SUCCESS)
            {
                devs = malloc(num_devs * sizeof(cl_device_id));
                err1 = clGetProgramInfo(*prog, CL_PROGRAM_DEVICES,
                    num_devs * sizeof(cl_device_id), devs, NULL);
                if (err1 == CL_SUCCESS)
                {
                    for (i = 0; i < num_devs; ++i)
                    {
                        cl_build_status status;

                        err1 = clGetProgramBuildInfo(*prog, devs[i],
                            CL_PROGRAM_BUILD_STATUS,
                            sizeof(cl_build_status), &status, NULL);
                        if ((err1 == CL_SUCCESS) && (status == CL_BUILD_ERROR))
                        {
                            char *log;
                            size_t log_size;

                            err1 = clGetProgramBuildInfo(*prog, devs[i],
                                CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                            if (err1 == CL_SUCCESS)
                            {
                                log = malloc(log_size * sizeof(char));
                                err1 = clGetProgramBuildInfo(*prog, devs[i],
                                    CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                                fprintf(stderr, "%s\n", log);
                                free(log);
                            }
                        }
                    }
                }

                free(devs);
            }
        }
        else
        {
            err = clCreateKernelsInProgram(*prog, 0, NULL, num_kerns);
            if (err == CL_SUCCESS)
            {
                *kerns = malloc((*num_kerns) * sizeof(cl_kernel));
                err = clCreateKernelsInProgram(*prog, *num_kerns, *kerns, NULL);
            }
        }
    }

    free(lens);
    return err;
}

cl_int opencl_kernels_from_files(cl_uint nfiles,
    const char **files, cl_context ctx, cl_program *prog,
    cl_uint *num_kerns, cl_kernel **kerns)
{
    cl_int err;

    size_t i;
    const char **frags;

    frags = malloc(nfiles * sizeof(char **));

    for (i = 0; i < nfiles; ++i)
    {
        FILE *fsrc;
        char *ssrc;
        size_t fsize;

        fsrc = fopen(files[i], "r");
        if (!fsrc) return CL_INVALID_VALUE;

        fseek(fsrc, 0, SEEK_END);
        fsize = ftell(fsrc);
        fseek(fsrc, 0, SEEK_SET);

        ssrc = malloc((fsize + 1) * sizeof(char));
        fread(ssrc, fsize, 1, fsrc);
        fclose(fsrc);

        frags[i] = ssrc;
    }

    err = opencl_kernels_from_fragments(nfiles, frags,
        ctx, prog, num_kerns, kerns);

    for (i = 0; i < nfiles; ++i)
    {
        free((void *) frags[i]);
    }

    free(frags);
    return err;
}

cl_int opencl_kernels_free_all(cl_uint *num_kerns, cl_kernel **kerns)
{
    cl_uint i;
    cl_int err;
    cl_kernel *k = *kerns;

    for (i = 0; i < *num_kerns; ++i)
    {
        err = clReleaseKernel(k[i]);
        if (err != CL_SUCCESS) return err;
        k[i] = NULL;
    }

    *kerns = NULL;
    *num_kerns = 0;
    return err;
}

cl_int opencl_kernel_lookup_by_name(cl_uint num_kerns,
    cl_kernel *kerns, const char *name, cl_kernel *kern)
{
    cl_uint i;
    char *kern_name;
    size_t len_kern_name, len_name;

    cl_int err;
    const cl_int err0 = CL_INVALID_KERNEL;

    err = err0;
    len_name = strlen(name) + 1;

    for (i = 0; i < num_kerns; ++i)
    {
        err = clGetKernelInfo(kerns[i], CL_KERNEL_FUNCTION_NAME,
            0, NULL, &len_kern_name);
        if ((err == CL_SUCCESS) && (len_kern_name == len_name))
        {
            kern_name = malloc(len_kern_name * sizeof(char));
            err = clGetKernelInfo(kerns[i], CL_KERNEL_FUNCTION_NAME,
                len_kern_name, kern_name, NULL);
            if ((err == CL_SUCCESS) &&
                (!strncmp(name, kern_name, len_name)))
            {
                *kern = kerns[i];
                free(kern_name);
                break;
            }

            free(kern_name);
        }

        err = err0;
    }

    return err;
}

cl_int opencl_program_free(cl_program *prog)
{
    cl_int err;
    err = clReleaseProgram(*prog);
    *prog = NULL;
    return err;
}
