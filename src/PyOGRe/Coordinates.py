from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Union

from IPython.display import display, Markdown
import sympy as sym

from PyOGRe.Options import options
from PyOGRe.Utils import zero_tensor
from PyOGRe.Exceptions import TransformationError, IndicesValueError
from PyOGRe.OGReObject import OGReObject, ValidateOGReObject

if TYPE_CHECKING:
    from PyOGRe.Metric import Metric


__doc__ = """
Coordinates Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


class Coordinates(OGReObject):
    """
    (Class) Coordinates(name, components, indices, symbol, transformations)

    Creates a new tensor object representing a coordinate system.

    `name`: Defines name of the tensor is used when displaying the coordinates.
    `components`: An array of the coordinate symbols.
    `indices`: Defines the indices of the coordinates, must be (1,).
    `symbol`: Defines the symbol used to represent the coordinates.
    `transformations`: Defines the coordinate transformations, but can be left as none and added later.

    Example:

    >>> import sympy as sym
    >>> from PyOGRe import new_coordinates
    >>> new_coordinates("Cartesian", sym.Array([sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")]))

    This will create a coordinates object named Cartesian with coordinate symbols x, y, and z.
    """

    def __init__(
        self: Coordinates,
        name: str,
        components: sym.Array,
        indices: Tuple[int, ...] = (1,),
        symbol: Union[str, sym.Symbol] = sym.Symbol('x'),
        transformations: Optional[Dict[Coordinates, List[sym.Eq]]] = None
    ) -> None:
        """
        Coordinates Constructor. Calls OGReObject constructor, then adds additional Coordinates
        specific attributes. Validates all user input to ensure that the Tensor is valid.
        """

        # Run the Coordinates Validation process
        ValidateCoordinates(
            name=name,
            components=components,
            indices=indices,
            symbol=symbol,
            transformations=transformations
        )

        # Call OGReObject constructor
        super().__init__(
            name=name,
            components=components,
            symbol=symbol,
            indices=indices
        )

        # Initialize the remaining attributes
        self._components[self] = {(1,): components}
        self._transformations: Dict[Coordinates, List[sym.Eq]] = {}
        if transformations:
            self._transformations = transformations
        self._jacobians: Dict[Coordinates, sym.Array] = {}
        self._inverse_jacobians: Dict[Coordinates, sym.Array] = {}
        self._christoffel_jacobians: Dict[Coordinates, sym.Array] = {}

    # ===================
    # Dunder Methods
    # ===================

    def __call__(
        self: Coordinates,
        indices: str = ""
    ) -> None:
        """
        """

        raise NotImplementedError(
            "Cannot perform Tensor calculations with Coordinate objects."
        )

    # ===================
    # Private Methods
    # ===================

    def _coord_functions(
        self: Coordinates,
        parameter: Optional[sym.Symbol] = None
    ) -> Dict[sym.Expr, sym.Expr]:
        parameter = options.CURVE_PARAMETER if parameter is None else parameter
        parameter_dot = sym.Symbol(str(parameter) + "dot")
        parameter_ddot = sym.Symbol(str(parameter) + "ddot")

        coordinates = self.get_components()
        f_coordinates = sym.Array(
            [
                sym.Function(str(coord))(parameter)
                if coord != parameter
                else coord
                for coord in coordinates
            ]
        )
        coordinates_dot = sym.Array(
            [
                sym.symbols(str(coord) + "dot")
                for coord in coordinates
            ]
        )
        f_coordinates_dot = sym.Array(
            [
                sym.Function(str(coord_dot))(parameter)
                if coord_dot != parameter_dot
                else coord_dot
                for coord_dot in coordinates_dot
            ]
        )
        coordinates_ddot = sym.Array(
            [
                sym.symbols(str(coord) + "ddot")
                for coord in coordinates
            ]
        )
        f_coordinates_ddot = sym.Array(
            [
                sym.Function(str(coord_ddot))(parameter)
                if coord_ddot != parameter_ddot
                else coord_ddot
                for coord_ddot in coordinates_ddot
            ]
        )

        return {
            **{
                coord_ddot: f_coord_ddot
                for coord_ddot, f_coord_ddot in zip(coordinates_ddot, f_coordinates_ddot)
            },
            **{
                coord_dot: f_coord_dot
                for coord_dot, f_coord_dot in zip(coordinates_dot, f_coordinates_dot)
            },
            **{
                coord: f_coord
                for coord, f_coord in zip(coordinates, f_coordinates)
            }
        }

    def _inverse_coord_functions(
        self: Coordinates,
        parameter: Optional[sym.Symbol] = None
    ) -> Dict[sym.Expr, sym.Expr]:
        parameter = options.CURVE_PARAMETER if parameter is None else parameter

        coordinates = self.get_components()
        coordinates_dot = sym.Array(
            [
                sym.symbols(str(coord) + "dot")
                for coord in coordinates
            ]
        )
        coordinates_ddot = sym.Array(
            [
                sym.symbols(str(coord) + "ddot")
                for coord in coordinates
            ]
        )

        subs = self._coord_functions(parameter=parameter)

        f_coordinates = coordinates.subs(subs)  # type: ignore[union-attr]
        df_coordinates = f_coordinates.diff(parameter)
        ddf_coordinates = df_coordinates.diff(parameter)

        return {
            **{
                ddf_coord: coord_ddot
                if ddf_coord != 0
                else None
                for ddf_coord, coord_ddot in zip(ddf_coordinates, coordinates_ddot)
            },
            **{
                df_coord: coord_dot
                if df_coord != 1
                else None
                for df_coord, coord_dot in zip(df_coordinates, coordinates_dot)
            },
            **{
                f_coord: coord
                for f_coord, coord in zip(f_coordinates, coordinates)
            }
        }

    def _get_info(
        self: Coordinates
    ) -> Dict[str, str]:
        info = super()._get_info()

        if self._transformations:
            transformations = []
            for key in self._transformations:
                transformations.append(key._name)
            info["Coordinate Transformations"] = ", ".join(transformations)

        if OGReObject._instances:
            coordinates_for = []
            for instance in OGReObject._instances:
                if hasattr(instance, "_coordinates"):
                    if getattr(instance, "_coordinates") is self:
                        coordinates_for.append(instance._name)
            info["Default Coordinates For"] = ", ".join(coordinates_for)

        return info

    def _check_jacobians(
        self: Coordinates,
        coords: Coordinates
    ) -> bool:
        """
        """

        if coords in self._jacobians:
            return True

        if coords in self._transformations:
            self._calculate_jacobian(coords=coords)
            return True

        raise TransformationError(
            error="rules",
            info=(self._name, coords._name)
        )

    def _calculate_jacobian(
        self: Coordinates,
        coords: Coordinates
    ) -> None:
        """
        """

        # Retrieve the coordinates
        self_coordinates = self.get_components()
        other_coordinates = coords.get_components()

        # Retrieve the rules
        rules = self._transformations[coords]

        ordered_rules: List[sym.Eq] = []
        for coord in self_coordinates:
            exists = False
            for rule in rules:
                if rule is not None and rule.lhs == coord:
                    exists = True
                    ordered_rules.append(rule)
            if not exists:
                ordered_rules.append(None)

        # Construct an empty array to store the Jacobian
        jacobian = zero_tensor(len(ordered_rules), 2)

        # Calculate the jacobian
        for i, rulei in enumerate(ordered_rules):
            for j in range(len(ordered_rules)):
                if rulei is None:
                    jacobian[i, i] = 1  # type: ignore[unreachable]
                else:
                    jacobian[i, j] = sym.diff(
                        rulei.rhs,
                        other_coordinates[j]
                    )

        # Construct an empty array to store the Christoffel Jacobian
        christoffel_jacobian = zero_tensor(len(ordered_rules), 3)

        # Calculate the Christoffel Jacobian
        for i, rulei in enumerate(ordered_rules):
            for j in range(len(ordered_rules)):
                for k in range(len(ordered_rules)):
                    if rulei is not None:
                        christoffel_jacobian[i, j, k] = sym.diff(
                            rulei.rhs,
                            other_coordinates[j],
                            other_coordinates[k]
                        )

        # Simplify, then add the Jacobian, inverse Jacobian, and Christoffel Jacobian to the object's dictionary
        jacobian = sym.Matrix(sym.simplify(jacobian))
        inverse_jacobian = sym.simplify(jacobian.inv())
        christoffel_jacobian = sym.Array(sym.simplify(christoffel_jacobian))

        self._jacobians[coords] = jacobian
        self._inverse_jacobians[coords] = inverse_jacobian
        self._christoffel_jacobians[coords] = christoffel_jacobian

    # ===================
    # Public Methods
    # ===================

    def get_components(
        self: Coordinates,
        mode: str = "sympy"
    ) -> Union[sym.Array, str]:
        """
        Returns the components in  the specified mode.
        """

        if not isinstance(mode, str):
            raise TypeError(
                f"Expected type '{str}' for argument 'mode', got '{type(mode)}'."
            )

        if mode.lower() not in ["sympy", "mathematica", "latex"]:
            raise ValueError(
                f"Expected value 'sympy', 'mathematica', or 'latex' for argument 'mode', got '{mode}'."
            )

        components = self._components[self][(1,)]

        if mode.lower() == "sympy":
            return components

        if mode.lower() == "mathematica":
            return str(sym.mathematica_code(components))

        if mode.lower() == "latex":
            return str(
                sym.latex(
                    components.reshape(self._components[self][(1,)].shape[0], 1)
                ).replace(r"\left[", r"\left(").replace(r"\right]", r"\right)")
            )

        return None

    def show(
        self: Coordinates
    ) -> None:
        """
        Shows the Coordinates symbolically.
        """

        if options.LATEX:
            size = options.FONT_SIZE
            representation_str = f"<div align=center style='font-size:{size}pt'> \n\n $$"
            representation_str += self._eq_symbol(indices=self._indices) + " = "
            representation_str += sym.latex(self._components[self][(1,)].reshape(self._components[self][(1,)].shape[0], 1))
            representation_str = representation_str.replace(r"\left[", r"\left(").replace(r"\right]", r"\right)")
            representation_str += "$$ \n\n </div>"
            display(Markdown(
                f"<div align=center style='font-size:{size+2}pt;margin-bottom:14pt'> \n\n" +
                f"{self._name}:" +
                "\n\n </div>" +
                representation_str
            ))

        elif not options.LATEX:
            representation_eq = sym.Eq(
                sym.Symbol(self._eq_symbol(indices=self._indices)),
                self._components[self][(1,)].reshape(self._components[self][(1,)].shape[0], 1)
            )
            sym.pprint(f"{self._name}:")
            sym.pprint(representation_eq)

    def list(
        self: Coordinates
    ) -> None:
        """
        Lists the Coordinates element wise.
        """

        if options.LATEX:
            size = options.FONT_SIZE
            display(Markdown(
                f"<div align=center style='font-size:{size+2}pt'> \n\n" +
                f"{self._name}:" +
                "\n\n </div>" +
                f"<div align=center style='font-size:{size}pt'> \n\n $$ \n" +
                "\\begin{aligned} \n" +
                r"\\[5pt]".join(
                    [
                        sym.latex(sym.Symbol(f"{self._symbol}^{symbol}")) + " &= " + sym.latex(symbol)
                        for symbol in self._components[self][(1,)]
                    ]
                ) +
                "\n \\end{aligned} \n $$ \n\n </div>"
            ))

        if not options.LATEX:
            sym.pprint(f"{self._name}:")
            for i in range(len(self._components[self][(1,)])):
                symbol = sym.Symbol(f"{self._symbol}^{self._components[self][(1,)][i]}")
                sym.pprint(sym.Eq(symbol, self._components[self][(1,)][i]))

    def add_coord_transformation(
        self: Coordinates,
        coords: Coordinates,
        rules: Optional[List[sym.Eq]] = None
    ) -> Coordinates:
        """
        (method) add_coord_transformation(coords, rules)

        Define a Coordinate transformation.

        `coords`: The target coordinate system.
        `rules`: The transformation rules to convert the current coordinates to the target coordinates.
        If left as None, and coords is supplied, any existing transformation will be removed.

        Example:

        >>> import sympy as sym
        >>> from PyOGRe.Coordinates import new_coordinates

        >>> t, x, y, z, r, theta, phi = sym.symbols("t x y z r theta phi")

        >>> cartesian = new_coordinates(
            name="4D Cartesian",
            components=sym.Array([t, x, y, z])
        )

        >>> spherical = new_coordinates(
            name="4D Spherical",
            components=sym.Array([t, r, theta, phi])
        )

        >>> cartesian.add_coord_transformation(
            coords=spherical,
            rules=[
                None,
                sym.Eq(x, r*sym.sin(theta)*sym.cos(phi)),
                sym.Eq(y, r*sym.sin(theta)*sym.sin(phi)),
                sym.Eq(z, r*sym.cos(theta))
            ]
        )

        >>> spherical.add_coord_transformation(
            coords=cartesian,
            rules=[
                None,
                sym.Eq(r, sym.sqrt(x**2+y**2+z**2)),
                sym.Eq(theta, sym.acos(z/(sym.sqrt(x**2+y**2+z**2)))),
                sym.Eq(phi, sym.atan(y/x)),
            ]
        )

        Above, we defined both the Cartesian and Spherical coordinate systems in 4D. We then
        added the coordinate transformations between the two systems.
        """

        if not isinstance(coords, Coordinates):
            raise TransformationError(
                error="type",
                info=type(coords)
            )

        if self == coords:
            raise TransformationError(
                error="coords",
                info="None"
            )

        if rules is None:
            del self._transformations[coords]
            return self

        for rule in rules:
            if rule is not None and not isinstance(rule, sym.Eq):
                raise TypeError(
                    f"Expected type 'List[sym.Eq]' for argument 'rules', got '{type(rule)}'."
                )

        self._transformations[coords] = rules

        return self

    def new_metric(
        self: Coordinates,
        name: str,
        components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
        indices: Tuple[int, int] = (-1, -1),
        symbol: Union[str, sym.Symbol] = sym.Symbol('g')
    ) -> Metric:
        """
        (method) new_metric(name, components, indices, symbol)

        Creates a new tensor object representing a metric tensor.

        `name`: Defines name of the tensor is used when displaying the metric.
        `components``: An array / matrix of the metric's components.
        `indices`: Defines the default indices of the metric, must be (1, 1) or (-1, -1).
        `symbol`: Defines the symbol used to represent the metric object.

        Example:

        >>> import sympy as sym
        >>> from PyOGRe import new_coordinates
        >>> cartesian = new_coordinates("Cartesian", sym.Array([sym.Symbol("t"), sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")]))
        >>> cartesian.new_metric("Minkowski", sym.diag(-1, 1, 1, 1))

        This will create the standard 3+1 dimensional Minkowski metric defined in standard Cartesian coordinates.
        """

        from PyOGRe.Metric import Metric

        return Metric(
            name=name,
            components=components,
            coords=self,
            indices=indices,
            symbol=symbol
        )


# Mapping class -> Ensures syntax similar to Mathematica version works
def new_coordinates(
    name: str,
    components: sym.Array,
    indices: Tuple[int, ...] = (1,),
    symbol: Union[str, sym.Symbol] = sym.Symbol('x'),
    transformations: Optional[Dict[Coordinates, List[sym.Eq]]] = None
) -> Coordinates:
    """
    (function) new_coordinates(name, components, indices, symbol, transformations)

    Creates a new tensor object representing a coordinate system.

    `name`: Defines name of the tensor is used when displaying the coordinates.
    `components`: An array of the coordinate symbols.
    `indices`: Defines the indices of the coordinates, must be (1,).
    `symbol`: Defines the symbol used to represent the coordinates.
    `transformations`: Defines the coordinate transformations, but can be left as none and added later.

    Example:

    >>> import sympy as sym
    >>> from PyOGRe import new_coordinates
    >>> new_coordinates("Cartesian", sym.Array([sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")]))

    This will create a coordinates object named Cartesian with coordinate symbols x, y, and z.
    """

    return Coordinates(
        name=name,
        components=components,
        indices=indices,
        symbol=symbol,
        transformations=transformations
    )


def ValidateCoordinates(
    name: str,
    components: sym.Array,
    indices: Optional[Tuple[int, ...]],
    symbol: Optional[Union[str, sym.Symbol]],
    transformations: Optional[Dict[Coordinates, List[sym.Eq]]]
) -> bool:
    """
    Validates a Coordinates object.
    """

    ValidateOGReObject(
        name=name,
        components=components,
        symbol=symbol,
        indices=indices
    )

    if indices is not None and indices != (1,):
        raise IndicesValueError(
            indices=indices,
            error="coordinates"
        )

    # If the transformation rules are supplied at initialization,
    # Check to see if they are valid
    if indices is not None and transformations:
        for rules in transformations.values():
            for rule in rules:
                if rule is not None and not isinstance(rule, sym.Eq):
                    raise TypeError(
                        f"Expected type 'List[sym.Eq]' for argument 'rules', got '{type(rule)}'."
                    )

    # Make sure Coordinates are supplied as a rank 1 SymPy.Array
    if components.rank() != 1:
        raise ValueError(
            f"Coordinates must be a rank 1 Tensor, not rank {components.rank()}."
        )

    return True
