import numpy
import functools
from itertools import chain, product

import coffee.base as coffee

import gem
from gem.optimise import remove_componenttensors as prune
from gem.utils import cached_property

from finat import TensorFiniteElement

import ufl

from tsfc.kernel_interface.common import KernelBuilderBase
from tsfc.finatinterface import create_element as _create_element
from tsfc.coffee import SCALAR_TYPE


# UFC DoF ordering for vector/tensor elements is XXXX YYYY ZZZZ.
create_element = functools.partial(_create_element, shape_innermost=False)


class KernelBuilder(KernelBuilderBase):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_type, subdomain_id, domain_number):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(integral_type.startswith("interior_facet"))
        self.integral_type = integral_type

        self.local_tensor = None
        self.coordinates_args = None
        self.coefficient_args = None
        self.coefficient_split = None

        if self.interior_facet:
            self._cell_orientations = (gem.Variable("cell_orientation_0", ()),
                                       gem.Variable("cell_orientation_1", ()))
        else:
            self._cell_orientations = (gem.Variable("cell_orientation", ()),)

        if integral_type == "exterior_facet":
            self._entity_number = {None: gem.VariableIndex(gem.Variable("facet", ()))}
        elif integral_type == "interior_facet":
            self._entity_number = {
                '+': gem.VariableIndex(gem.Variable("facet_0", ())),
                '-': gem.VariableIndex(gem.Variable("facet_1", ()))
            }
        elif integral_type == "vertex":
            self._entity_number = {None: gem.VariableIndex(gem.Variable("vertex", ()))}

    def set_arguments(self, arguments, multiindices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg multiindices: GEM argument multiindices
        :returns: GEM expression representing the return variable
        """
        self.local_tensor, prepare, expressions = prepare_arguments(
            arguments, multiindices, interior_facet=self.interior_facet)
        self.apply_glue(prepare)
        return expressions

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.Domain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        self.coordinates_args, expression = prepare_coordinates(
            f, "coordinate_dofs", interior_facet=self.interior_facet)
        self.coefficient_map[f] = expression

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        name = "w"
        self.coefficient_args = [
            coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                        pointers=[("const",), ()],
                        qualifiers=["const"])
        ]

        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        for n in range(len(integral_data.enabled_coefficients)):
            if not integral_data.enabled_coefficients[n]:
                continue

            coefficient = form_data.function_replace_map[form_data.reduced_coefficients[n]]
            expression = prepare_coefficient(coefficient, n, name, self.interior_facet)
            self.coefficient_map[coefficient] = expression

    def construct_kernel(self, name, body):
        """Construct a fully built kernel function.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg body: function body (:class:`coffee.Block` node)
        :returns: a COFFEE function definition object
        """
        args = [self.local_tensor]
        args.extend(self.coefficient_args)
        args.extend(self.coordinates_args)

        # Facet and vertex number(s)
        if self.integral_type == "exterior_facet":
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet")))
        elif self.integral_type == "interior_facet":
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet_0")))
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet_1")))
        elif self.integral_type == "vertex":
            args.append(coffee.Decl("std::size_t", coffee.Symbol("vertex")))

        # Cell orientation(s)
        if self.interior_facet:
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation_0")))
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation_1")))
        else:
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation")))

        return KernelBuilderBase.construct_kernel(self, name, args, body)

    def construct_empty_kernel(self, name):
        """Construct an empty kernel function.

        Kernel will just zero the return buffer and do nothing else.

        :arg name: function name
        :returns: a COFFEE function definition object
        """
        body = coffee.Block([])  # empty block
        return self.construct_kernel(name, body)

    @staticmethod
    def require_cell_orientations():
        # Nothing to do
        pass

    @staticmethod
    def needs_cell_orientations(ir):
        # UFC tabulate_tensor always have cell orientations
        return True

    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""
        return create_element(element, **kwargs)

    @cached_property
    def unsummed_coefficient_indices(self):
        return frozenset((ci,))


def prepare_coefficient(coefficient, num, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficient: UFL Coefficient
    :arg num: coefficient index in the original form
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: GEM expression referring to the Coefficient value
    """
    varexp = gem.Variable(name, (None, None, None))

    if coefficient.ufl_element().family() == 'Real':
        shape = coefficient.ufl_shape
        size = numpy.prod(shape, dtype=int)
        data = gem.view(varexp, slice(num, num + 1), slice(size), slice(ci_shape[0]))
        expression = gem.reshape(data, (), shape, ci_shape)
        expression = free_cell_index(expression)
        return expression

    element = create_element(coefficient.ufl_element())
    shape = element.index_shape
    size = numpy.prod(shape, dtype=int)

    def expression(data):
        result, = prune([gem.reshape(gem.view(data, slice(size), slice(ci_shape[0])), shape, ci_shape)])
        return result

    if not interior_facet:
        data = gem.view(varexp, slice(num, num + 1), slice(size), slice(ci_shape[0]))
        expression = expression(gem.reshape(data, (), (size,), ci_shape))
    else:
        data_p = gem.view(varexp, slice(num, num + 1), slice(size))
        data_m = gem.view(varexp, slice(num, num + 1), slice(size, 2 * size))
        expression = (expression(gem.reshape(data_p, (), (size,))),
                      expression(gem.reshape(data_m, (), (size,))))
    expression = free_cell_index(expression)
    return expression


def prepare_coordinates(coefficient, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    coordinates.

    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg     - :class:`coffee.Decl` function argument
         expression - GEM expression referring to the Coefficient
                      values
    """
    finat_element = create_element(coefficient.ufl_element())
    shape = finat_element.index_shape + ci_shape
    size = numpy.prod(shape, dtype=int)

    assert isinstance(finat_element, TensorFiniteElement)
    scalar_shape = finat_element.base_element.index_shape
    tensor_shape = finat_element._shape
    transposed_shape = scalar_shape + tensor_shape + ci_shape
    scalar_rank = len(scalar_shape)

    def transpose(expr):
        indices = tuple(gem.Index(extent=extent) for extent in expr.shape)
        transposed_indices = indices[scalar_rank:-1] + indices[:scalar_rank] + indices[-1:]
        return gem.ComponentTensor(gem.Indexed(expr, indices),
                                   transposed_indices)

    if not interior_facet:
        funargs = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                               pointers=[("",)],
                               qualifiers=["const"])]
        variable = gem.Variable(name, (size,))
        expression = transpose(gem.reshape(variable, transposed_shape))

        expression = free_cell_index(expression)
    else:
        funargs = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name+"_0"),
                               pointers=[("",)],
                               qualifiers=["const"]),
                   coffee.Decl(SCALAR_TYPE, coffee.Symbol(name+"_1"),
                               pointers=[("",)],
                               qualifiers=["const"])]
        variable0 = gem.Variable(name+"_0", (size,))
        variable1 = gem.Variable(name+"_1", (size,))
        expression = (transpose(gem.reshape(variable0, transposed_shape)),
                      transpose(gem.reshape(variable1, transposed_shape)))

        expression = map(free_cell_index, expression)

    return funargs, expression


def prepare_arguments(arguments, multiindices, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg multiindices: Argument multiindices
    :arg interior_facet: interior facet integral?
    :returns: (funarg, prepare, expressions)
         funarg      - :class:`coffee.Decl` function argument
         prepare     - list of COFFEE nodes to be prepended to the
                       kernel body
         expressions - GEM expressions referring to the argument
                       tensor
    """
    funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A"), pointers=[()])
    varexp = gem.Variable("A", (None,))

    #if len(arguments) == 0:
    #    # No arguments
    #    zero = coffee.FlatBlock(
    #        "memset({name}, 0, sizeof(*{name}));\n".format(name=funarg.sym.gencode())
    #    )
    #    return funarg, [zero], [gem.reshape(varexp, ())]

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    shapes = [element.index_shape for element in elements] + [ci_shape]
    indices = tuple(chain(*multiindices, (ci,)))


    def expression(restricted):
        return gem.Indexed(gem.reshape(restricted, *shapes), indices)

    u_shape = numpy.array([numpy.prod(element.index_shape, dtype=int)
                           for element in elements] + list(ci_shape))
    if interior_facet:
        c_shape = tuple(2 * u_shape)
        slicez = [[slice(r * s, (r + 1) * s)
                   for r, s in zip(restrictions, u_shape)]
                  for restrictions in product((0, 1), repeat=len(arguments))]
    else:
        c_shape = tuple(u_shape)
        slicez = [[slice(s) for s in u_shape]]

    expressions = [expression(gem.view(gem.reshape(varexp, c_shape), *slices))
                   for slices in slicez]

    zero = coffee.FlatBlock(
        str.format("memset({name}, 0, {size} * sizeof(*{name}));\n",
                   name=funarg.sym.gencode(), size=numpy.product(c_shape, dtype=int))
    )
    return funarg, [zero], prune(expressions)


ci_shape = (4,)
ci = gem.Index(extent=ci_shape[0])
def add_cell_index(expr):
    indices = tuple(gem.Index() for i in range(len(expr.shape)))
    return gem.ComponentTensor(gem.Indexed(expr, indices + (ci,)), indices)
def free_cell_index(expr):
    indices = tuple(gem.Index() for i in range(len(expr.shape[:-1])))
    return gem.ComponentTensor(gem.Indexed(expr, indices + (ci,)), indices)
