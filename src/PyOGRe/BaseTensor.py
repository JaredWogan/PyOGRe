from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import sympy as sym
from IPython.display import Markdown, display

from PyOGRe.Coordinates import Coordinates
from PyOGRe.Exceptions import TensorCoordinatesError, TransformationError
from PyOGRe.OGReObject import OGReObject, ValidateIndices, ValidateOGReObject
from PyOGRe.Options import options
from PyOGRe.Utils import map_to_array

if TYPE_CHECKING:
    from PyOGRe.Calc import CalcObject


__doc__ = """
BaseTensor Module: Abstract Base Class

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


class BaseTensor(OGReObject, ABC):
    """
    Base Tensor Class. This is an abstract class from which all other Tensor classes inherit.
    """

    def __init__(
        self: BaseTensor,
        name: str,
        components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
        coords: Coordinates,
        indices: Optional[Tuple[int, ...]] = None,
        symbol: Union[str, sym.Symbol] = sym.Symbol("T")
    ) -> None:
        """
        Tensor Constructor. Calls OGReObject constructor, then adds additional Tensor specific
        attributes. Validates all user input to ensure that the Tensor is valid.
        """

        # Check that coordinates is an instance of Coordinates
        if not isinstance(coords, Coordinates):
            raise TypeError(
                f"Expected type '{Coordinates}' for argument 'coordinates', got '{type(coords)}'"
            )

        # If the components are a matrix, convert them to an array
        if isinstance(components, sym.Matrix):
            components = sym.Array(components)

        # Run the BaseTensor Validation process
        ValidateBaseTensor(
            name=name,
            components=components,
            coordinates_components=coords._components[coords][(1,)],
            indices=indices,
            symbol=symbol
        )

        # Call OGReObject constructor
        super().__init__(
            name=name,
            components=components,
            symbol=symbol,
            indices=indices,
        )

        # Initialize the remaining attributes
        self._coordinates = coords
        self._components = {}
        self._components[self._coordinates] = {self._indices: components}

    # ===================
    # Private Methods
    # ===================

    def _get_representation(
        self: BaseTensor,
        coords: Coordinates,
        indices: Tuple[int, ...]
    ) -> sym.Array:
        """
        """

        if not isinstance(coords, Coordinates):
            raise TypeError(
                f"Expected type '{Coordinates}' for argument 'coords', got '{type(coords)}'."
            )

        if not isinstance(indices, tuple):
            raise TypeError(
                f"Expected type '{tuple}' for argument 'indices', got '{type(indices)}'."
            )

        ValidateIndices(indices=indices, source_indices=self._indices)

        # Check if the representation of the Tensor in the coords Coordinates has already been calculated
        if not self._check_representation(coords=coords, indices=indices):
            self._transform_coordinates(coords=coords, indices=indices)

        return self._components[coords][indices]

    def _transform_coordinates(
        self: BaseTensor,
        coords: Coordinates,
        indices: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        """
        if indices is None:
            indices = self._indices

        # Check to see if the Jacobians for the coords Coordinates have already been calculated
        self._coordinates._check_jacobians(coords=coords)

        # Retrieve the needed Jacobian, Inverse Jacobian, the transformation rules, and the required components
        components = self._components[self._coordinates][indices]
        jacobian = self._coordinates._jacobians[coords]
        inverse_jacobian = self._coordinates._inverse_jacobians[coords]
        rules = self._coordinates._transformations[coords]

        # Create a temporary list of Jacobians
        # For each index
        # if index is -1, add a factor of the Jacobian
        # if index is +1, add a factor of the inverse Jacobian
        # Additionally, determine the indices which will need to be contracted
        to_contract = []
        contraction_indices = []
        transformed_tensor = components
        for i, index in enumerate(indices):
            if index == 1:
                to_contract.append(inverse_jacobian)
                contraction_indices.append(
                    (i, self._rank + 2 * i + 1)
                )
            if index == -1:
                to_contract.append(jacobian)
                contraction_indices.append(
                    (i, self._rank + 2 * i)
                )

        # Perform a tensor product of all the Jacobians plus the components
        transformed_tensor = sym.tensorproduct(components, *to_contract)

        # Then, contract the tensor product with the list of contraction indices
        transformed_tensor = sym.tensorcontraction(
            transformed_tensor,
            *contraction_indices
        )

        # Once computed, go through and substitute the transformation rules
        # In order to simplify the expression in the new Coordinate system
        for rule in rules:
            if isinstance(rule, sym.Eq):
                transformed_tensor = transformed_tensor.subs({rule.lhs: rule.rhs})

        # Check if the destination coordinates are already in the dictionary
        # If not, create a new dictionary for the coords coordinates
        if coords not in self._components:
            self._components[coords] = {}

        # Store the transformed tensor in the dictionary after simplifying
        transformed_tensor = sym.simplify(transformed_tensor)
        self._components[coords][indices] = transformed_tensor

    def _get_info(
        self: BaseTensor
    ) -> Dict[str, str]:
        info = super()._get_info()
        info["Default Coordinates"] = self._coordinates._name
        return info

    # ===================
    # Public Methods
    # ===================

    def show(
        self: BaseTensor,
        coords: Optional[Coordinates] = None,
        indices: Optional[Tuple[int, ...]] = None,
        replace: Optional[Dict[Union[sym.Symbol, sym.Expr], Union[sym.Symbol, sym.Expr]]] = None,
        function: Optional[Callable[[Any], Any]] = None,
        args: Tuple[Any, ...] = ()
    ) -> None:
        """
        (method) show(coords, indices, replace, function, args)

        Shows the tensor as an array.

        `coords`: Display the repesentation in the given coordinates.
        `indices`: Display the repesentation with the given index configuration.
        `replace`: Dictionary of substitutions to be made.
        `function`: Function to be applied to each element of representation.
        `args`: Arguments to be passed to the function.
        """

        if coords is None:
            coords = self._coordinates

        if indices is None:
            indices = self._indices

        components = self._get_representation(coords=coords, indices=indices)
        if self._rank == 1:
            components = components.reshape(
                components.shape[0],
                1
            )

        if function is not None:
            components = map_to_array(
                components,
                function,
                *args
            )

        if replace is not None:
            components = components.subs(replace)
            components = components.doit()

        if options.LATEX:
            size = options.FONT_SIZE
            representation_str = f"<div align=center style='font-size:{size}pt'> \n\n $$"
            representation_str += self._eq_symbol(indices=indices) + r"\left("
            representation_str += ", ".join(
                list(
                    map(
                        sym.latex,
                        coords._components[coords][(1,)]
                    )
                )
            )
            representation_str += r"\right) =" + sym.latex(components) + "$$ \n\n </div>"
            representation_str = representation_str.replace(
                r"\left[", r"\left("
            ).replace(
                r"\right]", r"\right)"
            )
            if r"\frac" in representation_str:
                representation_str = representation_str.replace(r"\\", r"\\[1em]")

            display(Markdown(
                f"<div align=center style='font-size:{size+2}pt;margin-bottom:12pt'> \n\n" +
                f"{self._name}:" +
                "\n\n </div>" +
                representation_str
            ))

        if not options.LATEX:
            representation_eq = sym.Eq(
                sym.Function(sym.Symbol(self._eq_symbol(indices=indices)))(*coords._components[coords][(1,)]),
                components,
                evaluate=False
            )

            sym.pprint(f"{self._name}:")
            sym.pprint(representation_eq)

    def list(
        self: BaseTensor,
        coords: Optional[Coordinates] = None,
        indices: Optional[Tuple[int, ...]] = None,
        replace: Optional[Dict[Union[sym.Symbol, sym.Expr], Union[sym.Symbol, sym.Expr]]] = None,
        function: Optional[Callable[[Any], Any]] = None,
        args: Tuple[Any, ...] = ()
    ) -> None:
        """
        (method) list(coords, indices, replace, function, args)

        Lists the non-zero elements of the tensor.

        `coords`: Display the repesentation in the given coordinates.
        `indices`: Display the repesentation with the given index configuration.
        `replace`: Dictionary of substitutions to be made.
        `function`: Function to be applied to each element of representation.
        `args`: Arguments to be passed to the function.
        """

        if coords is None:
            coords = self._coordinates

        if indices is None:
            indices = self._indices

        components = self._get_representation(coords=coords, indices=indices)

        if function is not None:
            components = map_to_array(
                components,
                function,
                *args
            )

        if replace is not None:
            components = components.subs(replace)
            components = components.doit()

        non_zero: Dict[sym.Expr, List[Union[str, sym.Symbol]]] = {}

        permutations = product((i for i in range(self._coordinates._components[self._coordinates][(1,)].shape[0])), repeat=self._rank)

        if options.LATEX and self._rank > 0:
            for position in permutations:
                if components[position] != 0:
                    symbol = self._eq_symbol(indices=())
                    for i, index in enumerate(indices):
                        if index == -1:
                            symbol += "_{" + sym.latex(coords._components[coords][(1,)][position[i]]) + "}{ }"
                        if index == 1:
                            symbol += "^{" + sym.latex(coords._components[coords][(1,)][position[i]]) + "}{ }"

                    new_entry = True
                    for key in non_zero:
                        if key - components[position] == 0:
                            non_zero[key].append(symbol)
                            new_entry = False

                        if key + components[position] == 0:
                            non_zero[key].append("-" + symbol)
                            new_entry = False
                    if new_entry:
                        non_zero[components[position]] = [symbol]

        if not options.LATEX and self._rank > 0:
            for position in permutations:
                if components[position] != 0:
                    symbol = str(self._eq_symbol(indices=()))
                    for i, index in enumerate(indices):
                        if index == -1:
                            symbol += "_" + str(coords._components[coords][(1,)][position[i]])
                        if index == 1:
                            symbol += "^" + str(coords._components[coords][(1,)][position[i]])

                    new_entry = True
                    for key in non_zero:
                        if key - components[position] == 0:
                            non_zero[key].append(sym.Symbol(symbol))
                            new_entry = False

                        if key + components[position] == 0:
                            non_zero[key].append(sym.Symbol("-" + symbol))
                            new_entry = False
                    if new_entry:
                        non_zero[components[position]] = [sym.Symbol(symbol)]

        if self._rank == 0:
            if options.LATEX:
                non_zero = {components: [self._eq_symbol(indices=indices)]}
            if not options.LATEX:
                non_zero = {components: [sym.Symbol(self._eq_symbol(indices=indices))]}

        if options.LATEX:
            size = options.FONT_SIZE
            if non_zero:
                num = options.LIST_PER_LINE
                display(Markdown(
                    f"<div align=center style='font-size:{size+2}pt;margin-bottom:12pt'> \n\n" +
                    f"{self._name}:" +
                    "\n\n </div>" +
                    f"<div align=center style='font-size:{size}pt'> \n\n $$ \n" +
                    "\\begin{aligned} \n" +
                    (r"\\[10pt]").join(
                        [
                            "".join(
                                [
                                    " = ".join(value[slice(num*i, num*i+num) if num*i+num < len(value) else slice(num*i, len(value))])  # type: ignore[arg-type]
                                    + (" &= " + sym.latex(key) if num*i+num >= len(value) else r"= \\[2pt]")
                                    for i in range(ceil(len(value) / num))
                                ]
                            )
                            for key, value in non_zero.items()
                        ]
                    ) +
                    "\n \\end{aligned} \n $$ \n\n </div>"
                ))
            if not non_zero:
                display(Markdown(
                    f"<div align=center style='font-size:{size+2}pt;margin-bottom:12pt'> \n\n" +
                    f"{self._name}:" +
                    "\n\n </div>" +
                    f"<div align=center style='font-size:{size}pt'> No Non-Zero Elements </div>"
                ))

        if not options.LATEX:
            sym.pprint(f"{self._name}:")
            if non_zero:
                for key, item in iter(non_zero.items()):
                    for it in item:
                        sym.pprint(sym.Eq(it, key, evaluate=False))
                        print("\n")
            if not non_zero:
                sym.pprint("No Non-Zero Elements")

    def calc_norm_squared(
        self: BaseTensor
    ) -> BaseTensor:
        """
        Calculates the norm squared of the BaseTensor.
        """
        from PyOGRe.Calc import CalcNormSquared

        return CalcNormSquared(self)

    def change_default_indices(
        self: BaseTensor,
        indices: Tuple[int, ...]
    ) -> BaseTensor:
        """
        Changes the default indices of the Tensor.
        """

        # Make sure requested indices are valid.
        ValidateIndices(
            indices=indices,
            source_indices=self._indices
        )

        if self._check_representation(coords=self._coordinates, indices=indices):
            self._indices = indices

        return self

    def change_default_coords(
        self: BaseTensor,
        coords: Coordinates
    ) -> BaseTensor:
        """
        Changes the default Coordinates of the Tensor.
        """

        if not isinstance(coords, Coordinates):
            raise TypeError(
                f"Expected type '{Coordinates}' for argument 'coords', got '{type(coords)}'."
            )

        if coords is self._coordinates:
            return self

        if not self._coordinates._transformations or coords not in self._coordinates._transformations:
            raise TransformationError(
                error="rules",
                info=(self._coordinates._name, coords._name)
            )

        if not self._check_representation(coords=coords, indices=self._indices):
            self._transform_coordinates(coords=coords, indices=self._indices)

        self._coordinates = coords

        return self

    def get_components(  # type: ignore[override]
        self: BaseTensor,
        coords: Optional[Coordinates] = None,
        indices: Optional[Tuple[int, ...]] = None,
        replace: Optional[Dict[Union[sym.Symbol, sym.Expr], Union[sym.Symbol, sym.Expr]]] = None,
        function: Optional[Callable[[Any], Any]] = None,
        args: Tuple[Any, ...] = (),
        mode: str = "sympy"
    ) -> Union[sym.Array, str]:
        """
        Returns the components of the BaseTensor in the requested representation.
        """

        if not isinstance(mode, str):
            raise TypeError(
                f"Expected type '{str}' for argument 'mode', got '{type(mode)}'."
            )

        if mode.lower() not in ["sympy", "mathematica", "latex"]:
            raise ValueError(
                f"Expected value 'sympy', 'mathematica', or 'latex' for argument 'mode', got '{mode}'."
            )

        if coords is None:
            coords = self._coordinates

        if indices is None:
            indices = self._indices

        components = self._get_representation(coords=coords, indices=indices)

        if function is not None:
            components = map_to_array(
                components,
                function,
                *args
            )

        if replace is not None:
            components = components.subs(replace)
            components = components.doit()

        if not isinstance(components, sym.Array):
            components = sym.Array(components)  # type: ignore[unreachable]

        if mode.lower() == "sympy":
            return components

        if mode.lower() == "mathematica":
            if components.rank() == 0:
                components = components[()]
            return str(sym.mathematica_code(components))

        if mode.lower() == "latex":
            return str(sym.latex(components).replace(r"\left[", r"\left(").replace(r"\right]", r"\right)"))

        return None

    def simplify(
        self: BaseTensor
    ) -> BaseTensor:
        """
        Simplify the Tensor.
        """
        return simplify(self)

    # ===================
    # Abstract Methods
    # ===================

    @abstractmethod
    def _check_representation(
        self: BaseTensor,
        coords: Coordinates,
        indices: Tuple[int, ...]
    ) -> bool:
        """
        """
        return False

    @abstractmethod
    def __call__(
        self: BaseTensor,
        indices: str = ""
    ) -> CalcObject:
        """
        """

        from PyOGRe.Calc import CalcObject

        if not isinstance(indices, str):
            raise TypeError(
                f"Expected type '{str}' for argument 'indices', got '{type(indices)}'."
            )

        return CalcObject(tensor=self, indices=indices)


def ValidateBaseTensor(
    name: str,
    components: sym.Array,
    coordinates_components: sym.Array,
    indices: Optional[Tuple[int, ...]],
    symbol: Optional[Union[str, sym.Symbol]]
) -> bool:
    """
    Validates a BaseTensor object.
    """

    ValidateOGReObject(
        name=name,
        components=components,
        symbol=symbol,
        indices=indices
    )

    for i in range(components.rank()):
        if components.shape[i] != len(coordinates_components):
            raise TensorCoordinatesError(
                tensor_shape=components.shape,
                coordinates_shape=coordinates_components.shape
            )

    return True


def simplify(
    tensor: BaseTensor
) -> BaseTensor:
    """
    Simplifies the Tensor.
    """

    if options.LATEX:
        from tqdm.notebook import tqdm as progress
    else:
        from tqdm import tqdm as progress

    num_configurations = len(
        [indices for coords in tensor._components.keys() for indices in tensor._components[coords].keys()]
    )
    rank = tensor.rank()
    dim = len(
        tensor._coordinates.get_components()
    )

    # def index_simplify(
    #     index: Tuple[int, ...],
    #     component: sym.Expr
    # ) -> Tuple[Tuple[int, ...], sym.Expr]:
    #     return index, sym.simplify(component)

    with progress(
        total=num_configurations*dim**rank,
        desc="Simplifying: ",
        leave=False
    ) as pbar:
        for coords in tensor._components.keys():
            for indices in tensor._components[coords].keys():
                # # If parallelization is enabled, use multiprocessing
                # if options.PARALLEL:

                #     # For each configuration of the tensor, retrieve the components
                #     components = tensor._components[coords][indices]
                #     elements_components = elements(components)
                #     result_components = zero_tensor(dim=dim, rank=rank)

                #     # Retrieve the view
                #     view = options.VIEW

                #     async_result = view.map_async(
                #         index_simplify,
                #         elements_components.keys(),
                #         elements_components.values()
                #     )

                #     result = async_result.get()

                # # If parallelization is disabled, just simplify the components
                # else:
                if rank == 0:
                    tensor._components[coords][indices] = sym.simplify(sym.Array(tensor._components[coords][indices]))[0]
                if rank > 0:
                    tensor._components[coords][indices] = sym.simplify(tensor._components[coords][indices])

                pbar.update(dim**rank)

    return tensor
