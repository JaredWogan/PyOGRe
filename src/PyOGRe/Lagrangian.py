from __future__ import annotations

from itertools import product
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy as sym
from IPython.display import Markdown, display

from PyOGRe.Coordinates import Coordinates
from PyOGRe.Metric import Metric
from PyOGRe.Options import options
from PyOGRe.Tensor import Tensor
from PyOGRe.Utils import map_to_array

__doc__ = """
Lagrangian Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


class Lagrangian(Tensor):
    """
    Lagrangian Class. Derived from Tensor class, adds specialized transformation rules.
    """

    def __init__(
        self: Lagrangian,
        name: str,
        components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
        metric: Metric,
        coords: Coordinates,
        indices: Optional[Tuple[int, int, int]] = (1, -1, -1),
        symbol: Union[str, sym.Symbol] = sym.Symbol("L")
    ) -> None:
        """
        Christoffel Constructor. Calls Tensor constructor.
        Validates all user input to ensure that the Christoffel Tensor is valid.
        """

        # Call Tensor constructor
        super().__init__(
            name=name,
            components=components,
            metric=metric,
            coords=coords,
            indices=indices,
            symbol=symbol
        )

    # ===================
    # Private Methods
    # ===================

    def _transform_coordinates(
        self: Lagrangian,
        coords: Coordinates,
        indices: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Specialized transform_coordinates method for Lagrangian 'Tensors'.
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

        # For Lagrangians, we need to transform the derivatives with respect to the curve parameter
        parameter = options.CURVE_PARAMETER
        coordinates = coords.get_components()
        f_coordinates = sym.Array(
            [
                sym.Function(str(coord))(parameter)
                for coord in coordinates
            ]
        )
        df_coordinates = sym.Array(
            [
                coord.diff(parameter)
                for coord in f_coordinates
            ]
        )
        coordinates_dot = sym.Array(
            [
                sym.symbols(str(coord) + "dot")
                for coord in coordinates
            ]
        )

        # Once computed, go through and substitute the transformation rules
        # In order to simplify the expression in the new Coordinate system
        for rule in rules:
            if rule is not None and isinstance(rule, sym.Eq):
                symbol = rule.lhs
                substitute = rule.rhs

                symbol_dot = sym.symbols(str(symbol) + "dot")
                f_substitute = substitute.subs(
                    {
                        coord: f_coord
                        for coord, f_coord in zip(coordinates, f_coordinates)
                    }
                )
                df_substitute = f_substitute.diff(parameter)
                substitute_dot = df_substitute.subs(
                    {
                        **{
                            df_coord: coord_dot
                            for df_coord, coord_dot in zip(df_coordinates, coordinates_dot)
                        },
                        **{
                            f_coord: coord
                            for f_coord, coord in zip(f_coordinates, coordinates)
                        }
                    }
                )
                transformed_tensor = transformed_tensor.subs(
                    {
                        **{symbol: substitute},
                        **{symbol_dot: substitute_dot}
                    }
                )

        # Check if the destination coordinates are already in the dictionary
        # If not, create a new dictionary for the coords coordinates
        if coords not in self._components:
            self._components[coords] = {}

        # Store the transformed tensor in the dictionary after simplifying
        transformed_tensor = sym.simplify(transformed_tensor)
        self._components[coords][indices] = transformed_tensor

    # ===================
    # Public Methods
    # ===================

    def show(
        self: Lagrangian,
        coords: Optional[Coordinates] = None,
        indices: Optional[Tuple[int, ...]] = None,
        replace: Optional[Dict[Union[sym.Symbol, sym.Expr], Union[sym.Symbol, sym.Expr]]] = None,
        function: Optional[Callable[[Any], Any]] = None,
        args: Tuple[Any, ...] = ()
    ) -> None:
        """
        Shows the BaseTensor components symbollically.
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
            parameter = options.CURVE_PARAMETER

            subs = coords._coord_functions()
            inverse_subs = coords._inverse_coord_functions()

            f_substitute = sym.Array(list(replace.items())).subs(subs)
            df_substitute = f_substitute.diff(parameter)
            ddf_substitute = df_substitute.diff(parameter)

            f_subs, df_subs, ddf_subs = f_substitute.subs(inverse_subs), df_substitute.subs(inverse_subs), ddf_substitute.subs(inverse_subs)

            components = components.subs(
                {
                    **{
                        f_sub[0]: f_sub[1]
                        for f_sub in f_subs
                    },
                    **{
                        df_sub[0]: df_sub[1]
                        for df_sub in df_subs
                    },
                    **{
                        ddf_sub[0]: ddf_sub[1]
                        for ddf_sub in ddf_subs
                    }
                }
            ).doit()

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
        self: Lagrangian,
        coords: Optional[Coordinates] = None,
        indices: Optional[Tuple[int, ...]] = None,
        replace: Optional[Dict[Union[sym.Symbol, sym.Expr], Union[sym.Symbol, sym.Expr]]] = None,
        function: Optional[Callable[[Any], Any]] = None,
        args: Tuple[Any, ...] = ()
    ) -> None:
        """
        Lists the BaseTensor components element wise.
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
            parameter = options.CURVE_PARAMETER

            subs = coords._coord_functions()
            inverse_subs = coords._inverse_coord_functions()

            f_substitute = sym.Array(list(replace.items())).subs(subs)
            df_substitute = f_substitute.diff(parameter)
            ddf_substitute = df_substitute.diff(parameter)

            f_subs, df_subs, ddf_subs = f_substitute.subs(inverse_subs), df_substitute.subs(inverse_subs), ddf_substitute.subs(inverse_subs)

            components = components.subs(
                {
                    **{
                        f_sub[0]: f_sub[1]
                        for f_sub in f_subs
                    },
                    **{
                        df_sub[0]: df_sub[1]
                        for df_sub in df_subs
                    },
                    **{
                        ddf_sub[0]: ddf_sub[1]
                        for ddf_sub in ddf_subs
                    }
                }
            ).doit()

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

    def get_components(  # type: ignore[override]
        self: Lagrangian,
        coords: Optional[Coordinates] = None,
        indices: Optional[Tuple[int, ...]] = None,
        replace: Optional[Dict[Union[sym.Symbol, sym.Expr], Union[sym.Symbol, sym.Expr]]] = None,
        function: Optional[Callable[[Any], Any]] = None,
        args: Tuple[Any, ...] = (),
        mode: str = "sympy"
    ) -> Union[sym.Array, str]:
        """
        Returns the components of the Lagrangian in the requested representation.
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
            parameter = options.CURVE_PARAMETER

            subs = coords._coord_functions()
            inverse_subs = coords._inverse_coord_functions()

            f_substitute = sym.Array(list(replace.items())).subs(subs)
            df_substitute = f_substitute.diff(parameter)
            ddf_substitute = df_substitute.diff(parameter)

            f_subs, df_subs, ddf_subs = f_substitute.subs(inverse_subs), df_substitute.subs(inverse_subs), ddf_substitute.subs(inverse_subs)

            components = components.subs(
                {
                    **{
                        f_sub[0]: f_sub[1]
                        for f_sub in f_subs
                    },
                    **{
                        df_sub[0]: df_sub[1]
                        for df_sub in df_subs
                    },
                    **{
                        ddf_sub[0]: ddf_sub[1]
                        for ddf_sub in ddf_subs
                    }
                }
            )

        # Replace all coordinates with coordinate functions
        subs = coords._coord_functions()
        components = components.subs(subs)

        if not isinstance(components, sym.Array):
            components = sym.Array(components)  # type: ignore[unreachable]

        if mode.lower() == "sympy":
            return components

        if mode.lower() == "mathematica":
            return str(sym.mathematica_code(components))

        if mode.lower() == "latex":
            return str(sym.latex(components).replace(r"\left[", r"\left(").replace(r"\right]", r"\right)"))

        return None
