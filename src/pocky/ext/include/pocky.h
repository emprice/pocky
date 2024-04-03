/**
 * @file pocky.h
 * @brief Common definitions for the @c pocky Python extension
 */

#ifndef POCKY_H
#define POCKY_H

/** Standardizes the definition of @c size_t for Python extensions */
#define PY_SSIZE_T_CLEAN

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define CL_TARGET_OPENCL_VERSION    (300)

#include <CL/opencl.h>

/** Exception object for OpenCL-specific errors */
extern PyObject *pocky_ocl_error;

/** Handle of the Python @c Platform type */
extern PyTypeObject *pocky_platform_type;

/** Handle of the Python @c Device type */
extern PyTypeObject *pocky_device_type;

#include "pocky_context.h"
#include "pocky_bufpair.h"
#include "pocky_functions.h"
#include "pocky_helpers.h"
#include "pocky_utils.h"

#endif      /* POCKY_H */

/* vim: set ft=c.doxygen: */
