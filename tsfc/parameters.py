from __future__ import absolute_import, print_function, division

import numpy


NUMPY_TYPE = numpy.dtype("double")

SCALAR_TYPE = {numpy.dtype("double"): "double",
               numpy.dtype("float32"): "float"}[NUMPY_TYPE]


PARAMETERS = {
    "quadrature_rule": "auto",
    "quadrature_degree": "auto",

    # Maximum extent to unroll index sums. Default is 3, so that loops
    # over geometric dimensions are unrolled; this improves assembly
    # performance.  Can be disabled by setting it to None, False or 0;
    # that makes compilation time much shorter.
    "unroll_indexsum": 3,

    # Precision of float printing (number of digits)
    "precision": numpy.finfo(NUMPY_TYPE).precision,
}


def default_parameters():
    return PARAMETERS.copy()


def get_common_parameter(key, parameters, *metadata, defaults=[None]):
    """Extract value under given key from metadata, check it is
    the same in all provided metadatas, otherwise raise ValueError.
    If nothing beyond default is provided fetch it from parameters
    and eventually from global defaults. If it is still default, raise
    KeyError.

    :arg key: parameter name
    :arg parameters: parameters dictionary
    :arg *metadata: integral metadata
    :arg defaults: iterable of values considered equivalent to default
    :returns: non-default value, otherwise raise KeyError
    """
    # Pick single representation for defaults
    default = next(v for v in defaults)

    # Extract all values from metadata
    values = set(md.get(key, default) for md in metadata)

    # Translate defaults
    for v in values:
        if v in defaults:
            values.remove(v)
            values.add(default)

    if len(values) > 1:
        raise ValueError("Got multiple metadata values '%s' for key '%s' "
                         "where expected same" % (values, key))

    # Get the unique metadata value, parameters value, or global default
    value, = values or (default,)
    if value in defaults:
        value = parameters.get(key, PARAMETERS.get(key))

    if value in defaults:
        raise KeyError("Failed to extract parameter value for key '%s' "
                       "from provided metadata '%s', parameters '%s', "
                       "or defaults '%s'"
                       % (key, metadata, parameters, PARAMETERS))

    return value
