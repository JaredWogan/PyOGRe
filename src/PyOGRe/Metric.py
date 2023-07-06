from __future__ import annotations
import json

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import sympy as sym
from IPython.display import Markdown, display

from PyOGRe.BaseTensor import BaseTensor
from PyOGRe.Coordinates import Coordinates
from PyOGRe.Exceptions import IndicesDimensionError
from PyOGRe.OGReObject import OGReObject, ValidateOGReObject
from PyOGRe.Options import options

if TYPE_CHECKING:
    from PyOGRe.Calc import CalcObject
    from PyOGRe.Christoffel import Christoffel
    from PyOGRe.Geodesic import (GeodesicChristoffel, GeodesicLagrangian,
                                 GeodesicTime)
    from PyOGRe.Lagrangian import Lagrangian
    from PyOGRe.Tensor import Tensor


__doc__ = """
Metric Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


class Metric(BaseTensor):
    """
    (Class) Metric(name, components, coords, indices, symbol)

    Creates a new tensor object representing a metric tensor.

    `name`: Defines name of the tensor is used when displaying the metric.
    `components`: An array / matrix of the metric's components.
    `coords`: Defines the coordinates the metric is using.
    `indices`: Defines the default indices of the metric, must be (1, 1) or (-1, -1).
    `symbol`: Defines the symbol used to represent the metric object.

    Example:

    >>> import sympy as sym
    >>> from PyOGRe import new_coordinates
    >>> cartesian = new_coordinates("Cartesian", sym.Array([sym.Symbol("t"), sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")]))
    >>> cartesian.new_metric("Minkowski", sym.diag(-1, 1, 1, 1))

    This will create the standard 3+1 dimensional Minkowski metric defined in standard Cartesian coordinates.
    """

    def __init__(
        self: Metric,
        name: str,
        components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
        coords: Coordinates,
        indices: Tuple[int, int] = (-1, -1),
        symbol: Union[str, sym.Symbol] = sym.Symbol("g")
    ) -> None:
        """
        Metric Constructor. Calls Tensor constructor, then adds additional Metric specific
        attributes. Validates all user input to ensure that the Tensor is valid.
        """

        ValidateMetric(
            name=name,
            components=components,
            indices=indices,
            symbol=symbol
        )

        # Call the BaseTensor constructor
        super().__init__(
            name=name,
            components=sym.Array(components),
            coords=coords,
            indices=indices,
            symbol=symbol
        )

        self._metric = self
        # For a Metric, calculate all the representations in advance
        self._components[self._coordinates][(1, -1)] = sym.Array(
            sym.eye(self._components[self._coordinates][self._indices].shape[0])
        )
        self._components[self._coordinates][(-1, 1)] = sym.Array(
            sym.eye(self._components[self._coordinates][self._indices].shape[0])
        )
        self._components[self._coordinates][tuple(-i for i in indices)] = sym.Array(
            sym.Matrix(self._components[self._coordinates][self._indices]).inv()
        )

    # ===================
    # Dunder Methods
    # ===================

    def __call__(
        self: Metric,
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

    # ===================
    # Private Methods
    # ===================

    def _check_representation(
        self: Metric,
        coords: Coordinates,
        indices: Tuple[int, ...]
    ) -> bool:
        """
        """

        if coords in self._components:
            return True
        return False

    def _transform_coordinates(
        self: Metric,
        coords: Coordinates,
        indices: Optional[Tuple[int, ...]] = None
    ) -> None:
        super()._transform_coordinates(coords=coords, indices=indices)

        if indices is None:
            indices = self._indices

        self._components[coords][tuple(-i for i in indices)] = sym.Array(
            sym.Matrix(self._components[coords][self._indices]).inv()
        )

        self._components[coords][(indices[0], -indices[1])] = sym.Array(
            sym.tensorcontraction(
                sym.tensorproduct(
                    self._components[coords][tuple(-i for i in indices)],
                    self._components[coords][indices]
                ),
                (1, 3)
            )
            # sym.eye(self._components[self._coordinates][self._indices].shape[0])
        )

        self._components[coords][(-indices[0], indices[1])] = sym.Array(
            sym.tensorcontraction(
                sym.tensorproduct(
                    self._components[coords][tuple(-i for i in indices)],
                    self._components[coords][indices]
                ),
                (0, 2)
            )
            # sym.eye(self._components[self._coordinates][self._indices].shape[0])
        )

    def _get_info(
        self: Metric
    ) -> Dict[str, str]:
        info = super()._get_info()
        if OGReObject._instances:
            metric_for = []
            for instance in OGReObject._instances:
                if hasattr(instance, "_metric"):
                    if getattr(instance, "_metric") is self:
                        metric_for.append(instance._name)
            info["Tensors Using This Metric"] = ", ".join(metric_for)
        return info

    # ===================
    # Public Methods
    # ===================

    def line_element(
        self: Metric,
        coords: Optional[Coordinates] = None
    ) -> None:
        """
        (method) line_element(coords)

        Displays the line element of the Metric.

        `coords`: The coordinates to use when displaying the line element.
        Defaults to use the Metric's default coordinates.
        """

        if coords is not None and not isinstance(coords, Coordinates):
            raise TypeError(
                f"Expected type '{Coordinates}' for argument 'coords', got '{type(coords)}'."
            )

        if coords is None:
            coords = self._coordinates

        # If coords Coordinates is not specified, use the default Coordinates
        if not self._check_representation(coords=coords, indices=(-1, -1)):
            self._transform_coordinates(coords=coords, indices=(-1, -1))

        # Retrieve the components of the Metric
        components = self._components[coords][(-1, -1)]

        # Retrieve the symbols for the coords Coordinates
        symbols = coords._components[coords][(1,)]

        if options.LATEX:
            font_size = options.FONT_SIZE

            # Create the line element in LaTeX
            space_time_interval = 0
            d = sym.Symbol("d", commutative=False)
            for i in range(components.shape[0]):
                for j in range(components.shape[1]):
                    diff1 = sym.Symbol(str(symbols[i if i < j else j]), commutative=False)
                    diff2 = sym.Symbol(str(symbols[j if i < j else i]), commutative=False)
                    if diff1 == diff2:
                        space_time_interval += components[i, j] * d * diff1 * diff2
                    else:
                        space_time_interval += components[i, j] * d * diff1 * d * diff2
            space_time_interval_str = sym.latex(space_time_interval)

            # Add $ for inline math
            line_element = r"\mathrm{d}s^2" + " = " + space_time_interval_str
            for i in range(components.shape[0]):
                line_element = line_element.replace("d" + str(symbols[i]), r"\mathrm{d}" + str(symbols[i]))
            line_element = f"<div align=center style='font-size:{font_size}pt'> \n\n $$" + line_element + "$$ \n\n </div>"

            # Print the line element in Latex
            display(Markdown(
                f"<div align=center style='font-size:{font_size + 2}pt;margin-bottom:12pt'> \n\n {self._name}: \n\n </div>" +
                line_element
            ))

        if not options.LATEX:

            # Define the symbols to be used
            ds = sym.Symbol("ds")

            # Create infinitessimals
            infinitesimals = []
            for symbol in symbols:
                infinitesimals.append(
                    sym.Symbol("d" + str(symbol))
                )

            # Create the spacetime interval
            space_time_interval_eq = 0
            for i in range(components.shape[0]):
                for j in range(components.shape[1]):
                    space_time_interval_eq += components[i, j] * infinitesimals[i] * infinitesimals[j]

            line_element = sym.Eq(ds**2, space_time_interval_eq)

            # Print the line element
            sym.pprint(self._name)
            sym.pprint(line_element)

    def volume_element(
        self: Metric,
        coords: Optional[Coordinates] = None
    ) -> None:
        """
        (method) volume_element(coords)

        Displays the volume element squared of the Metric.

        `coords`: The coordinates to use when displaying the volume element.
        Defaults to use the Metric's default coordinates.
        """

        if coords is not None and not isinstance(coords, Coordinates):
            raise TypeError(
                f"Expected type '{Coordinates}' for argument 'coords', got '{type(coords)}'."
            )

        # If coords is not specified, use the default Coordinates
        if coords is None:
            coords = self._coordinates

        # Make sure the metric in the requested Coordinates exists
        if not self._check_representation(coords=coords, indices=(-1, -1)):
            self._transform_coordinates(coords=coords, indices=(-1, -1))

        # Retrieve the components of the Metric
        components = sym.Matrix(self._components[coords][(-1, -1)])

        volume_element_squared = sym.simplify(components.det())

        if options.LATEX:
            font_size = options.FONT_SIZE
            display(Markdown(
                f"<div align=center style='font-size:{font_size + 2}pt;margin-bottom:12pt'> \n\n {self._name}: \n\n </div>" +
                f"<div align=center style='font-size:{font_size}pt'> \n\n $$" +
                r"\mathrm{d}V^2 = " +
                sym.latex(volume_element_squared) +
                "$$ \n\n </div>"
            ))

        if not options.LATEX:
            sym.pprint(self._name)
            sym.pprint(volume_element_squared)

        return None

    def new_tensor(
        self: Metric,
        name: str,
        components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
        coords: Optional[Coordinates] = None,
        indices: Optional[Tuple[int, ...]] = None,
        symbol: Union[str, sym.Symbol] = sym.Symbol('T')
    ) -> Tensor:
        """
        (method) new_tensor(name, components, coords, indices, symbol)

        Creates a new tensor object representing a tensor.

        `name`: Defines name of the tensor is used when displaying the tensor.
        `components`: An array / matrix of the tensor's components.
        `coords`: Defines the coordinates the tensor is using.
        `indices`: Defines the default indices of the tensor, each index must be 1 (contravariant) or -1 (covariant).
        `symbol`: Defines the symbol used to represent the metric object.

        Example:

        >>> import sympy as sym
        >>> from PyOGRe import new_coordinates, new_metric, new_tensor
        >>> cartesian = new_coordinates("Cartesian", sym.Array([sym.Symbol("t"), sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")]))
        >>> minkowski = new_metric("Minkowski", sym.diag(-1, 1, 1, 1), cartesian)
        >>> minkowski.new_tensor("Scalar", sym.Array(42), symbol="S")

        This will create the scalar 42 in the Minkowski metric and Cartesian coordinates.
        """
        from PyOGRe.Tensor import Tensor

        return Tensor(
            name=name,
            components=components,
            metric=self,
            coords=coords if coords is not None else self._coordinates,
            indices=indices,
            symbol=symbol
        )

    def calc_christoffel(
        self: Metric
    ) -> Christoffel:
        """
        Calculates the Christoffel symbols from the metric.
        """
        from PyOGRe.Calc import CalcChristoffel

        return CalcChristoffel(self)

    def calc_riemann_tensor(
        self: Metric
    ) -> Tensor:
        """
        Calculates the Riemann Tensor from the metric.
        """
        from PyOGRe.Calc import CalcRiemannTensor

        return CalcRiemannTensor(self)

    def calc_ricci_tensor(
        self: Metric
    ) -> Tensor:
        """
        Calculates the Ricci Tensor from the metric.
        """
        from PyOGRe.Calc import CalcRicciTensor

        return CalcRicciTensor(self)

    def calc_ricci_scalar(
        self: Metric
    ) -> Tensor:
        """
        Calculates the Ricci Scalar from the metric.
        """
        from PyOGRe.Calc import CalcRicciScalar

        return CalcRicciScalar(self)

    def calc_einstein_tensor(
        self: Metric
    ) -> Tensor:
        """
        Calculates the Einstein Tensor from the metric.
        """
        from PyOGRe.Calc import CalcEinsteinTensor

        return CalcEinsteinTensor(self)

    def calc_weyl_tensor(
        self: Metric
    ) -> Tensor:
        """
        Calculates the Weyl Tensor from the metric.
        """
        from PyOGRe.Calc import CalcWeylTensor

        return CalcWeylTensor(self)

    def calc_lagrangian(
        self: Metric,
        coords: Optional[Coordinates] = None
    ) -> Lagrangian:
        """
        Calculates the Lagrangian from the metric.
        """
        from PyOGRe.Calc import CalcLagrangian

        return CalcLagrangian(metric=self, coords=coords)

    def calc_geodesic_from_lagrangian(
        self: Metric,
        coords: Optional[Coordinates] = None,
        activate: bool = True
    ) -> GeodesicLagrangian:
        """
        Calculates the Geodesic from the metric.
        """
        from PyOGRe.Calc import CalcGeodesicFromLagrangian

        return CalcGeodesicFromLagrangian(metric=self, coords=coords, activate=activate)

    def calc_geodesic_from_christoffel(
        self: Metric,
        coords: Optional[Coordinates] = None
    ) -> GeodesicChristoffel:
        """
        Calculates the Geodesic from the metric.
        """
        from PyOGRe.Calc import CalcGeodesicFromChristoffel

        return CalcGeodesicFromChristoffel(metric=self, coords=coords)

    def calc_geodesic_with_time_parameter(
        self: Metric,
        coords: Optional[Coordinates] = None
    ) -> GeodesicTime:
        """
        Calculates the Geodesic from the metric, with respect to the time coordinate.
        """
        from PyOGRe.Calc import CalcGeodesicWithTimeParameter

        return CalcGeodesicWithTimeParameter(metric=self, coords=coords)


# Mapping function -> Ensures syntax similar to Mathematica version works
def new_metric(
    name: str,
    components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
    coords: Coordinates,
    indices: Tuple[int, int] = (-1, -1),
    symbol: Union[str, sym.Symbol] = sym.Symbol('g')
) -> Metric:
    """
    (function) new_metric(name, components, coords, indices, symbol)

    Creates a new tensor object representing a metric tensor.

    `name`: Defines name of the tensor is used when displaying the metric.
    `components`: An array / matrix of the metric's components.
    `coords`: Defines the coordinates the metric is using.
    `indices`: Defines the default indices of the metric, must be (1, 1) or (-1, -1).
    `symbol`: Defines the symbol used to represent the metric object.

    Example:

    >>> import sympy as sym
    >>> from PyOGRe import new_coordinates, new_metric
    >>> cartesian = new_coordinates("Cartesian", sym.Array([sym.Symbol("t"), sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")]))
    >>> new_metric("Minkowski", sym.diag(-1, 1, 1, 1), cartesian)

    This will create the standard 3+1 dimensional Minkowski metric defined in standard Cartesian coordinates.
    """

    return Metric(
        name=name,
        components=components,
        coords=coords,
        indices=indices,
        symbol=symbol
    )


def ValidateMetric(
    name: str,
    components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
    indices: Tuple[int, int],
    symbol: Union[str, sym.Symbol]
) -> bool:
    """
    Validates a Metric object.
    """

    if not isinstance(components, sym.Matrix) and not isinstance(components, sym.Array):
        raise TypeError(
            f"Expected type '{sym.Matrix}' or '{sym.Array}' for argument 'components', got '{type(components)}'."
        )

    ValidateOGReObject(
        name=name,
        components=sym.Array(components),
        symbol=symbol,
        indices=indices,
    )

    if sym.Matrix(components).det() == 0 or not sym.Matrix(components).is_symmetric():
        raise ValueError(
            "The Metric must be invertible and symmetric."
        )

    if indices in [(1, -1), (-1, 1)]:
        raise ValueError(
            f"Cannot define a metric with indice configuration {indices}. Argument 'indices' must be either (-1, -1) or (1, 1) to define a Metric."
        )

    if indices is not None and len(indices) != 2:
        raise IndicesDimensionError(
            indices=indices,
            error="metric",
            dim=2
        )

    return True
