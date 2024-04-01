#ifndef POCKY_HELPERS_H
#define POCKY_HELPERS_H

/** Docstring for this Python extension */
extern const char pocky_module_doc[];

/** Docstring for the Python @c Platform type */
extern const char platform_type_doc[];

/** Docstring for the Python @c Device type */
extern const char device_type_doc[];

/** Docstring for the Python @c Context type */
extern const char context_type_doc[];

/** Docstring for the Python @c BufferPair type */
extern const char bufpair_type_doc[];

extern const char ocl_msg_not_a_device[];
extern const char ocl_msg_not_a_capsule[];
extern const char ocl_msg_not_a_valid_context[];
extern const char ocl_msg_kernel_not_found[];
extern const char ocl_fmt_internal[];

/** Fields available in the Python @c Platform type */
extern PyStructSequence_Field platform_fields[];

/** Struct sequence descriptor for the Python @c Platform type */
extern PyStructSequence_Desc platform_desc;

/** Fields available in the Python @c Device type */
extern PyStructSequence_Field device_fields[];

/** Struct sequence descriptor for the Python @c Device type */
extern PyStructSequence_Desc device_desc;

/** Fields available in the Python @c Context type */
extern PyStructSequence_Field context_fields[];

#endif      /* POCKY_HELPERS_H */
