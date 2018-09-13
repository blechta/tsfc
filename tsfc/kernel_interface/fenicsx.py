import numpy

import coffee.base as coffee

import gem
from gem.optimise import remove_componenttensors as prune

from tsfc.kernel_interface.ufc import KernelBuilder as UFCKernelBuilder
from tsfc.kernel_interface.ufc import create_element


class KernelBuilder(UFCKernelBuilder):
    """Helper class for building a :class:`Kernel` object."""

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        name = "w"
        self.coefficient_args = [
            coffee.Decl(self.scalar_type, coffee.Symbol(name),
                        pointers=[()],
                        qualifiers=["const"])
        ]

        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        offset = 0
        for n in range(len(integral_data.enabled_coefficients)):

            # Prepare also disabled coefficients to compute offset
            coefficient = form_data.function_replace_map[form_data.reduced_coefficients[n]]
            expression, offset = prepare_coefficient(coefficient, offset, name, self.interior_facet)

            if integral_data.enabled_coefficients[n]:
                self.coefficient_map[coefficient] = expression


def prepare_coefficient(coefficient, offset, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficient: UFL Coefficient
    :arg offset: starting flattened index
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (expression, offset)
         expression - GEM expression referring to the Coefficient value
         offset - terminating flattened index
    """
    varexp = gem.Variable(name, (None,))

    if coefficient.ufl_element().family() == 'Real':
        size = numpy.prod(coefficient.ufl_shape, dtype=int)
        data = gem.view(varexp, slice(offset, offset + size))
        offset += size
        return gem.reshape(data, coefficient.ufl_shape), offset

    element = create_element(coefficient.ufl_element())
    size = numpy.prod(element.index_shape, dtype=int)

    def expression(data):
        result, = prune([gem.reshape(gem.view(data, slice(size)), element.index_shape)])
        return result

    if not interior_facet:
        data = gem.view(varexp, slice(offset, offset + size))
        offset += size
        return expression(gem.reshape(data, (size,))), offset
    else:
        data_p = gem.view(varexp, slice(offset,        offset + size))
        data_m = gem.view(varexp, slice(offset + size, offset + 2*size))
        offset += 2*size
        return ((expression(gem.reshape(data_p, (size,))),
                 expression(gem.reshape(data_m, (size,)))),
                offset)
