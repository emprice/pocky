#include "pocky.h"

const char pocky_module_doc[] = "A Python bridge to OpenCL\n";

const char platform_type_doc[] =
    "Platform(id: Capsule, name: str, version: str)\n"
    "Encapsulated OpenCL platform ID with basic information.\n\n"
    "Args:\n"
    "  id: Unique platform ID\n"
    "  name: Platform name, as reported by the hardware\n"
    "  version: Platform version, as reported by the hardware\n\n"
    "Danger:\n"
    "  Do not instantiate this class directly! Rather, use valid output from "
    "  :py:func:`pocky.list_all_platforms()`.\n";

const char device_type_doc[] = "This is a stub";

const char context_type_doc[] =
    "Context()\n"
    "Opaque object encapsulating the OpenCL context, command queues, kernels, "
    "and program.\n\n"
    "Danger:\n"
    "  Do not instantiate this class directly! Rather, use valid output from "
    "  :py:func:`pocky.context_default()` or "
    "  :py:func:`pocky.context_from_devices()`.\n";

const char bufpair_type_doc[] =
    "BufferPair(context: Context, host_array: numpy.ndarray)\n"
    "Pair of buffers, one host and one device, used for input and output "
    "to compiled kernels. Copy operations are carried out before and after "
    "each kernel execution.\n\n"
    "Args:\n"
    "  context: Valid existing OpenCL context\n"
    "  host_array: NumPy array to serve as host memory buffer\n";

const char ocl_msg_not_a_device[] = "At least one list item was not a valid "
    "device descriptor; did you use output from list_all_devices() ?";
const char ocl_msg_not_a_capsule[] = "At least one list item was not a valid "
    "capsule object; did you use output from this module?";
const char ocl_msg_not_a_valid_context[] = "An invalid context was passed; "
    "did you use output from this module?";
const char ocl_msg_kernel_not_found[] = "The kernel could not be found "
    "among the compiled kernels.";
const char ocl_fmt_internal[] = "Internal OpenCL error occurred with code %s (%d)";

PyStructSequence_Field platform_fields[] = {
    { "id", NULL },         /* platform id */
    { "name", NULL },       /* platform name */
    { "version", NULL },    /* platform version */
    { NULL, NULL }          /* sentinel value */
};

PyStructSequence_Desc platform_desc = {
    "Platform",         /* struct sequence type name */
    platform_type_doc,  /* documentation for this type */
    platform_fields,    /* list of fields in this type */
    3                   /* number of visible fields */
};

PyStructSequence_Field device_fields[] = {
    { "id", NULL },         /* device id */
    { "name", NULL },       /* device name */
    { "types", NULL },      /* device types */
    { NULL, NULL }          /* sentinel value */
};

PyStructSequence_Desc device_desc = {
    "Device",           /* struct sequence type name */
    device_type_doc,    /* documentation for this type */
    device_fields,      /* list of fields in this type */
    3                   /* number of visible fields */
};

PyStructSequence_Field context_fields[] = {
    { "context", NULL },    /* context handle */
    { "queues", NULL },     /* command queue(s) */
    { "program", NULL },    /* compiled program */
    { "kernels", NULL },    /* available kernel(s) */
    { NULL, NULL }          /* sentinel value */
};

PyStructSequence_Desc context_desc = {
    "Context",          /* struct sequence type name */
    context_type_doc,   /* documentation for this type */
    context_fields,     /* list of fields in this type */
    2                   /* number of visible fields */
};

PyStructSequence_Field bufpair_fields[] = {
    { "host", NULL },   /* host buffer (NumPy array) */
    { "device", NULL }, /* device buffer handle */
    { NULL, NULL }      /* sentinel value */
};

PyStructSequence_Desc bufpair_desc = {
    "BufferPair",       /* struct sequence type name */
    bufpair_type_doc,   /* documentation for this type */
    bufpair_fields,     /* list of fields in this type */
    2                   /* number of visible fields */
};
