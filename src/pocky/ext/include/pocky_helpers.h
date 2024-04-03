#ifndef POCKY_HELPERS_H
#define POCKY_HELPERS_H

/** Docstring for this Python extension */
extern const char pocky_module_doc[];

/** Docstring for the Python @c pocky.ext.Platform type */
extern const char pocky_platform_type_doc[];

/** Docstring for the Python @c pocky.ext.Device type */
extern const char pocky_device_type_doc[];

/** Docstring for the Python @c pocky.ext.Context type */
extern const char pocky_context_type_doc[];

/** Docstring for the Python @c pocky.ext.BufferPair type */
extern const char pocky_bufpair_type_doc[];

extern const char pocky_ocl_msg_not_a_device[];
extern const char pocky_ocl_msg_not_a_capsule[];
extern const char pocky_ocl_msg_not_a_valid_context[];
extern const char pocky_ocl_msg_kernel_not_found[];
extern const char pocky_ocl_fmt_internal[];

/** Fields available in the Python @c pocky.ext.Platform type */
extern PyStructSequence_Field pocky_platform_fields[];

/** Struct sequence descriptor for the Python @c pocky.ext.Platform type */
extern PyStructSequence_Desc pocky_platform_desc;

/** Fields available in the Python @c pocky.ext.Device type */
extern PyStructSequence_Field pocky_device_fields[];

/** Struct sequence descriptor for the Python @c pocky.ext.Device type */
extern PyStructSequence_Desc pocky_device_desc;

/** Fields available in the Python @c pocky.ext.Context type */
extern PyStructSequence_Field pocky_context_fields[];

#endif      /* POCKY_HELPERS_H */

/* vim: set ft=c.doxygen: */
