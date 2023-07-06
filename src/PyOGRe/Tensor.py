from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import sympy as sym

from PyOGRe.BaseTensor import BaseTensor
from PyOGRe.Coordinates import Coordinates
from PyOGRe.Metric import Metric
from PyOGRe.Utils import contract

if TYPE_CHECKING:
    from PyOGRe.Calc import CalcObject


__doc__ = """
Tensor Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


class Tensor(BaseTensor):
    """
    (Class) new_tensor(name, components, metric, coords, indices, symbol)

    Creates a new tensor object representing a tensor.

    `name`: Defines name of the tensor is used when displaying the tensor.
    `components`: An array / matrix of the tensor's components.
    `metric`: Defines the metric tensor associated with the tensor.
    `coords`: Defines the coordinates the tensor is using.
    `indices`: Defines the default indices of the tensor, each index must be 1 (contravariant) or -1 (covariant).
    `symbol`: Defines the symbol used to represent the metric object.

    Example:

    >>> import sympy as sym
    >>> from PyOGRe import new_coordinates, new_metric, new_tensor
    >>> cartesian = new_coordinates("Cartesian", sym.Array([sym.Symbol("t"), sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")]))
    >>> minkowski = new_metric("Minkowski", sym.diag(-1, 1, 1, 1), cartesian)
    >>> og.new_tensor("Scalar", sym.Array(42), minkowski, cartesian, (), "S")

    This will create the scalar 42 in the Minkowski metric and Cartesian coordinates.
    """

    def __init__(
        self: Tensor,
        name: str,
        components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
        metric: Metric,
        coords: Coordinates,
        indices: Optional[Tuple[int, ...]] = None,
        symbol: Union[str, sym.Symbol] = sym.Symbol("T")
    ) -> None:
        """
        Tensor Constructor. Calls OGReObject constructor, then adds additional Tensor specific
        attributes. Validates all user input to ensure that the Tensor is valid.
        """

        # Check that metric is an instance of Metric
        if not isinstance(metric, Metric):
            raise TypeError(
                f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'"
            )

        # Call BaseTensor constructor
        super().__init__(
            name=name,
            components=components,
            coords=coords,
            symbol=symbol,
            indices=indices
        )

        # Initialize the remaining attributes
        self._metric: Metric = metric

    # ===================
    # Dunder Methods
    # ===================

    def __call__(
        self: Tensor,
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
        self: Tensor,
        coords: Coordinates,
        indices: Tuple[int, ...]
    ) -> bool:
        """
        """

        if coords in self._components:
            if indices in self._components[coords]:
                return True
            if indices not in self._components[coords]:
                self._raise_lower(coords=coords, indices=indices)
                return True

        return False

    def _raise_lower(
        self: Tensor,
        coords: Coordinates,
        indices: Tuple[int, ...]
    ) -> None:
        """
        """

        # Check to see if required metric representations exist
        if not self._metric._check_representation(coords=coords, indices=(-1, -1)):
            self._metric._transform_coordinates(coords=coords, indices=(-1, -1))

        # Retrieve the needed components, the metric and the inverse metric in the coords Coordinates
        source_indices, source_components = list(iter(self._components[coords].items()))[0]
        metric_target = self._metric._components[coords][(-1, -1)]
        inverse_metric_target = self._metric._components[coords][(1, 1)]

        # Create a temporary list of Jacobians
        # For each index
        # if index is -1, add a factor of the Jacobian
        # if index is +1, add a factor of the inverse Jacobian
        # Additionally, determine the indices which will need to be contracted
        to_contract = []
        contraction_indices = []
        for i, (initial_index, target_index) in enumerate(zip(source_indices, indices)):
            # If we need to raise an index, contract with the inverse metric
            if initial_index == -1 and target_index == 1:
                to_contract.append(inverse_metric_target)
                contraction_indices.append(
                    # (1, i + 2)
                    (1, i)
                )

            # If we need to lower an index, contract with the metric
            if initial_index == 1 and target_index == -1:
                to_contract.append(metric_target)
                contraction_indices.append(
                    # (1, i + 2)
                    (1, i)
                )

            # Otherwise, contract with the identiy matrix
            if initial_index == target_index:
                to_contract.append(sym.eye(source_components.shape[0]))
                contraction_indices.append(
                    # (1, i + 2)
                    (1, i)
                )

        # Raise and Lower the indices one at a time
        raise_lowered = source_components
        for i, (metric, index) in iter(enumerate(zip(to_contract, contraction_indices))):
            indice_order = list(range(len(raise_lowered.shape)))
            indice_order.remove(0)
            indice_order.insert(i, 0)
            raise_lowered = contract(
                metric,
                raise_lowered,
                index
            )
            raise_lowered = sym.permutedims(raise_lowered, indice_order)

        self._components[coords][indices] = sym.simplify(raise_lowered)

    def _transform_coordinates(
        self: Tensor,
        coords: Coordinates,
        indices: Optional[Tuple[int, ...]] = None
    ) -> None:

        if indices is None:
            indices = self._indices

        if not self._check_representation(coords=self._coordinates, indices=indices):
            self._raise_lower(coords=self._coordinates, indices=indices)

        super()._transform_coordinates(coords=coords, indices=indices)

    def _get_info(
        self: Tensor
    ) -> Dict[str, str]:
        info = super()._get_info()
        info["Metric"] = self._metric._name
        return info


# Mapping class -> Ensures syntax similar to Mathematica version works
def new_tensor(
    name: str,
    components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
    metric: Metric,
    coords: Coordinates,
    indices: Optional[Tuple[int, ...]] = None,
    symbol: Union[str, sym.Symbol] = sym.Symbol('T')
) -> Tensor:
    """
    (function) new_tensor(name, components, metric, coords, indices, symbol)

    Creates a new tensor object representing a tensor.

    `name`: Defines name of the tensor is used when displaying the tensor.
    `components`: An array / matrix of the tensor's components.
    `metric`: Defines the metric tensor associated with the tensor.
    `coords`: Defines the coordinates the tensor is using.
    `indices`: Defines the default indices of the tensor, each index must be 1 (contravariant) or -1 (covariant).
    `symbol`: Defines the symbol used to represent the metric object.

    Example:

    >>> import sympy as sym
    >>> from PyOGRe import new_coordinates, new_metric, new_tensor
    >>> cartesian = new_coordinates("Cartesian", sym.Array([sym.Symbol("t"), sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")]))
    >>> minkowski = new_metric("Minkowski", sym.diag(-1, 1, 1, 1), cartesian)
    >>> og.new_tensor("Scalar", sym.Array(42), minkowski, cartesian, (), "S")

    This will create the scalar 42 in the Minkowski metric and Cartesian coordinates.
    """

    return Tensor(
        name=name,
        components=components,
        metric=metric,
        coords=coords,
        indices=indices,
        symbol=symbol
    )
