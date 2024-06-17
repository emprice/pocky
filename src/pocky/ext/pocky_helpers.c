#include "pocky.h"

const char pocky_module_doc[] = "A Python bridge to OpenCL\n";

const char pocky_platform_type_doc[] =
    "Platform(id: Capsule, name: str, version: str)\n"
    "Encapsulated OpenCL platform ID with basic information.\n\n"
    "Args:\n"
    "  id: Unique platform ID\n"
    "  name: Platform name, as reported by the hardware\n"
    "  version: Platform version, as reported by the hardware\n\n"
    "Danger:\n"
    "  Do not instantiate this class directly! Rather, use valid output from "
    "  :py:func:`pocky.list_all_platforms()`.\n";

const char pocky_device_type_doc[] = "This is a stub";

const char pocky_context_type_doc[] =
    "Context()\n"
    "Opaque object encapsulating the OpenCL context, command queues, kernels, "
    "and program.\n\n"
    "Danger:\n"
    "  Do not instantiate this class directly! Rather, use valid output from "
    "  :py:func:`pocky.Context.default()` or "
    "  :py:func:`pocky.Context.from_device_list()`.\n";

const char pocky_bufpair_type_doc[] =
    "BufferPair(context: Context, host_array: numpy.ndarray)\n"
    "Pair of buffers, one host and one device, used for input and output "
    "to compiled kernels. Copy operations are carried out before and after "
    "each kernel execution.\n\n"
    "Args:\n"
    "  context: Valid existing OpenCL context\n"
    "  host_array: NumPy array to serve as host memory buffer\n";

const char pocky_ocl_msg_not_a_device[] = "At least one list item was not a "
    "valid device descriptor; did you use output from list_all_devices() ?";
const char pocky_ocl_msg_not_a_capsule[] = "At least one list item was not a "
    "valid capsule object; did you use output from this module?";
const char pocky_ocl_msg_not_a_valid_context[] = "An invalid context was "
    "passed; did you use output from this module?";
const char pocky_ocl_msg_kernel_not_found[] = "The kernel could not be found "
    "among the compiled kernels.";
const char pocky_ocl_fmt_internal[] = "Internal OpenCL error occurred with "
    "code %s (%d)";

PyStructSequence_Field pocky_platform_fields[] = {
    { "id", NULL },         /* platform id */
    { "name", NULL },       /* platform name */
    { "version", NULL },    /* platform version */
    { NULL, NULL }          /* sentinel value */
};

PyStructSequence_Desc pocky_platform_desc = {
    "Platform",                 /* struct sequence type name */
    pocky_platform_type_doc,    /* documentation for this type */
    pocky_platform_fields,      /* list of fields in this type */
    3                           /* number of visible fields */
};

PyStructSequence_Field pocky_device_fields[] = {
    { "id", NULL },         /* device id */
    { "name", NULL },       /* device name */
    { "types", NULL },      /* device types */
    { NULL, NULL }          /* sentinel value */
};

PyStructSequence_Desc pocky_device_desc = {
    "Device",                   /* struct sequence type name */
    pocky_device_type_doc,      /* documentation for this type */
    pocky_device_fields,        /* list of fields in this type */
    3                           /* number of visible fields */
};

/* vim: set ft=c.doxygen: */
