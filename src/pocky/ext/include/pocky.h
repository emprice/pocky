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

#include <CL/opencl.h>

/** Exception object for OpenCL-specific errors */
extern PyObject *ocl_error;

/** Handle of the Python @c Platform type */
extern PyTypeObject *platform_type;

/** Handle of the Python @c Device type */
extern PyTypeObject *device_type;

#include "context.h"
#include "bufpair.h"
#include "functions.h"
#include "helpers.h"

#endif      /* POCKY_H */
