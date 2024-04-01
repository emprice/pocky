#ifndef POCKY_FUNCTIONS_H
#define POCKY_FUNCTIONS_H

/**
 * @brief Module method to list all available platforms
 * @param[in] self Module instance reference
 * @param[in] args Python arguments to be parsed; for this function,
 * should be empty
 * @return Python @c PyList object containing a @c Platform struct
 * for each available platform
 */
extern PyObject *list_all_platforms(PyObject *self, PyObject *args);

/**
 * @brief Module method to list all available devices for a platform
 * @param[in] self Module instance reference
 * @param[in] args Python arguments to be parsed; expects exactly one
 * argument that must be a Python @c Platform struct
 * @return Python @c PyList object containing a @c Device struct
 * for each available device
 */
extern PyObject *list_all_devices(PyObject *self, PyObject *args);

#endif      /* POCKY_FUNCTIONS_H */
