from __future__ import annotations

import copy
from itertools import permutations
from typing import List, Optional, Tuple, Union

import sympy as sym
from PyOGRe.BaseTensor import BaseTensor

from PyOGRe.Christoffel import Christoffel
from PyOGRe.Coordinates import Coordinates
from PyOGRe.Geodesic import GeodesicLagrangian, GeodesicChristoffel, GeodesicTime
from PyOGRe.Lagrangian import Lagrangian
from PyOGRe.Metric import Metric
from PyOGRe.OGReObject import ValidateIndices
from PyOGRe.Options import options
from PyOGRe.Tensor import Tensor
from PyOGRe.Utils import contract, partial_contract

__doc__ = """
Calc Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


class CalcObject:
    """
    CalcObject Class.

    ====================

    Attributes:

    --------------------


    ====================

    Methods:

    --------------------

    """

    __slots__ = ["tensor", "indices", "role"]

    def __init__(
        self,
        tensor: Optional[Union[Metric, Tensor]] = None,
        indices: Union[str, Tuple[sym.Symbol, ...]] = "",
        role: str = "tensor"
    ) -> None:
        """
        CalcObject constructor. Stores the tensor along with the string of indices.
        Validates all user input.
        """

        if tensor is not None and not isinstance(tensor, (Metric, Tensor)):
            raise TypeError(
                f"Expected type '{Metric}' or type '{Tensor}' for argument 'tensor', got '{type(tensor)}'."
            )

        if not isinstance(indices, (str, tuple, sym.Symbol)):
            raise TypeError(
                f"Expected type '{str}', type '{tuple}', or type '{sym.Symbol}' for argument 'indices', got '{type(indices)}'."
            )

        if not isinstance(role, str):
            raise TypeError(
                f"Expected type '{str}' for argument 'derivative', got '{type(role)}'."
            )

        if isinstance(indices, str) and indices != "":
            indices = sym.symbols(indices, seq=True)

        if tensor is not None:
            ValidateIndices(
                indices=tensor._indices,
                components=tensor.get_components(),
            )

        if indices != "":
            for indice_symbol in indices:
                count = indices.count(indice_symbol)  # type: ignore[arg-type]
                # If an index is repeated more than twice, it is invalid
                if count > 2:
                    raise ValueError(
                        f"Index ({indice_symbol}) cannot be repeated more than twice (Repeated {count} times)."
                    )

                # Check to see if we need to calculate any traces
                if count == 2:
                    # Determine which indices need to be contracted
                    contraction_indices = tuple(
                        index
                        for index, value in enumerate(indices)
                        if value == indice_symbol
                    )

                    # Determine the correct configuration of the metric to contract with
                    metric_config = tuple(
                        -tensor._indices[index]
                        for index in contraction_indices
                    )
                    metric_components = tensor._metric.get_components(indices=metric_config)

                    # Calculate the trace
                    trace_components = contract(
                        metric_components,  # type: ignore[arg-type]
                        tensor.get_components(),  # type: ignore[arg-type]
                        (0, contraction_indices[0]),
                        (1, contraction_indices[1])
                    )

                    # Configure the new tensor
                    trace_indices = tuple(
                        index
                        for i, index in enumerate(tensor._indices)
                        if i not in contraction_indices
                    )
                    indices = tuple(
                        symbol  # type: ignore[misc]
                        for symbol in indices
                        if symbol != indice_symbol
                    )

                    # Update the tensor
                    tensor = Tensor(
                        name="TEMP",
                        components=trace_components,
                        metric=tensor._metric,
                        coords=tensor._coordinates,
                        indices=trace_indices,
                        symbol=tensor._symbol  # type: ignore[arg-type]
                    )

        self.tensor: Optional[Union[Metric, Tensor]] = tensor
        self.indices: Union[str, Tuple[sym.Symbol, ...]] = indices
        self.role: str = role

    # ===================
    # Dunder Methods
    # ===================

    def __repr__(
        self: CalcObject
    ) -> str:
        if self.role == "partial":
            return f"{self.role}=>PartialD[{self.indices}]"
        if self.role == "covariant":
            return f"{self.role}=>CovariantD[{self.indices}]"
        return f"{self.role}=>{self.tensor._name}[{self.indices}]"

    def __add__(
        self: CalcObject,
        other: object
    ) -> CalcObject:
        """
        """

        # If the other object is just the integer 0, return the orginal object
        if other == 0:
            return self

        # Make sure we are only adding / subtracting another CalcObject
        if not isinstance(other, CalcObject):
            raise TypeError(
                f"Expected type '{CalcObject}' for argument 'other', got '{type(other)}'."
            )

        # Make sure we are not adding partial or covariant derivatives
        if self.role != "tensor" or other.role != "tensor":
            raise ValueError(
                "Cannot add two partial or covariant derivatives."
            )

        # Make sure the metrics of both objects are the same
        if getattr(self.tensor, "_metric", None) != getattr(other.tensor, "_metric", None):
            raise ValueError(
                "Cannot add two objects with different Metric Tensors."
            )

        # Make sure the ranks of both tensors are the same
        if self.tensor.rank() != other.tensor.rank():
            raise ValueError(
                "Cannot add two Tensors with different ranks."
            )

        # Make sure the indices of both tensors are the same, up to permutation
        if other.indices and self.indices:
            if other.indices not in list(permutations(self.indices)):
                raise ValueError(
                    f"Indices {self.indices} and {other.indices} must be the same up to a permutation for Tensor addition."
                )

        # Enumerate the indices of the first tensor and indices positions
        if not isinstance(self.indices, sym.Symbol):  # type: ignore[unreachable]
            symbols_self = {
                symbol: i
                for i, symbol in enumerate(iter(self.indices))
            }
        if isinstance(self.indices, sym.Symbol):  # type: ignore[unreachable]
            symbols_self = {
                self.indices: 0
            }

        # Enumerate the indices of the second tensor and indices positions
        if not isinstance(other.indices, sym.Symbol):  # type: ignore[unreachable]
            symbols_other = {
                symbol: i
                for i, symbol in enumerate(iter(other.indices))
            }
        if isinstance(other.indices, sym.Symbol):  # type: ignore[unreachable]
            symbols_other = {
                other.indices: 0
            }

        # Retrieve the components of each tensor
        components_self = self.tensor.get_components(
            coords=self.tensor._coordinates,
            indices=self.tensor._indices
        )
        permutation = tuple(
            symbols_other[symbol]
            for symbol in symbols_self
        )
        inv_permutation = tuple(
            symbols_self[symbol]
            for symbol in symbols_other
        )
        other_indices = tuple(
            self.tensor._indices[i]
            for i in inv_permutation
        )
        components_other = other.tensor.get_components(
            coords=self.tensor._coordinates,
            indices=other_indices
        )

        # Permute the indices of the second tensor to match the first
        components_other = sym.permutedims(components_other, permutation)

        # Add the two tensors
        result_components = components_self + components_other

        if self.tensor._rank == 0:
            result = result_components

        if self.tensor._rank > 0:
            result = sym.Array(sym.simplify(result_components))

        result_tensor = Tensor(
            name="TEMP",
            components=result,
            metric=self.tensor._metric,
            coords=self.tensor._coordinates,
            indices=self.tensor._indices
        )

        return CalcObject(
            tensor=result_tensor,
            indices=self.indices,
        )

    def __radd__(
        self: CalcObject,
        other: object
    ) -> CalcObject:
        """
        """

        return self + other

    def __neg__(
        self: CalcObject
    ) -> CalcObject:
        """
        """
        # Make sure we are not negating a partial or covariant derivative
        if self.role != "tensor":
            raise ValueError(
                "Cannot negate a partial or covariant derivative."
            )

        # Simply negate the default components of the tensor, and return the result
        result_components = -self.tensor._components[self.tensor._coordinates][self.tensor._indices]

        result = Tensor(
            name="TEMP",
            components=result_components,
            metric=self.tensor._metric,
            coords=self.tensor._coordinates,
            indices=self.tensor._indices
        )

        return CalcObject(
            tensor=result,
            indices=self.indices,
        )

    def __sub__(
        self: CalcObject,
        other: object
    ) -> CalcObject:
        """
        """

        # If the other object is just the integer 0, return the orginal object
        if other == 0:
            return self

        # Make sure we are only adding / subtracting another CalcObject
        if not isinstance(other, CalcObject):
            raise TypeError(
                f"Expected type '{CalcObject}' for argument 'other', got '{type(other)}'."
            )

        # Make sure we are not subtracting partial or covariant derivatives
        if self.role != "tensor" or other.role != "tensor":
            raise ValueError(
                "Cannot subtract two partial or covariant derivatives."
            )

        # Make sure the metrics of both objects are the same
        if getattr(self, "_metric", None) != getattr(other, "_metric", None):
            raise ValueError(
                "Cannot subtract two objects with different Metric Tensors."
            )

        # Make sure the ranks of both tensors are the same
        if self.tensor._rank != other.tensor._rank:
            raise ValueError(
                "Cannot subtract two Tensors with different ranks."
            )

        # Make sure the indices of both tensors are the same, up to permutation
        if self.indices and other.indices:
            if other.indices not in list(permutations(self.indices)):
                raise ValueError(
                    f"Indices {self.indices} and {other.indices} must be the same up to a permutation for Tensor subtraction."
                )

        return self + (-other)

    def __rsub__(
        self: CalcObject,
        other: object
    ) -> CalcObject:
        """
        """

        return -self

    def __mul__(
        self: CalcObject,
        other: object
    ) -> CalcObject:
        """
        """

        # Make sure we are not multiplying a partial or covariant derivative
        if self.role != "tensor":
            raise ValueError(
                "Cannot multiply a partial or covariant derivative."
            )

        # Make sure we are only multiplying by a scalar
        if not isinstance(other, (int, float, sym.Expr, sym.Symbol, CalcObject)):
            raise TypeError(
                f"Expected type '{int}', type '{float}', type '{sym.Symbol}', or type '{sym.Expr}' for argument 'other', got '{type(other)}'."
            )

        if isinstance(other, CalcObject) and other.tensor._rank > 0 and self.tensor._rank > 0:
            raise ValueError(
                "Cannot multiply a tensor by a tensor, use tensor contraction instead."
            )

        # If the other is a scalar, simply multiply the components of the tensor
        if not isinstance(other, CalcObject):
            # Convert float to sympy objects
            other = sym.nsimplify(other)

            # Simply multiple all the components by the scalar
            result_components = other * self.tensor._components[self.tensor._coordinates][self.tensor._indices]
            result_metric = self.tensor._metric
            result_coordinates = self.tensor._coordinates
            result_indices = self.tensor._indices
            result_calc_indices = self.indices

        # If the other is a tensor (scalar), multiply the components of the tensor
        if isinstance(other, CalcObject):
            # Determine which CalcObject is the scalar
            if self.tensor._rank == 0:
                scalar = self.tensor.get_components(coords=other.tensor._coordinates)[()]  # type: ignore[index]
                result_components = (
                    scalar *
                    other.tensor._components[other.tensor._coordinates][other.tensor._indices]
                )
                result_metric = other.tensor._metric
                result_coordinates = other.tensor._coordinates
                result_indices = other.tensor._indices
                result_calc_indices = other.indices

            if other.tensor._rank == 0:
                scalar = other.tensor.get_components(coords=self.tensor._coordinates)[()]  # type: ignore[index]
                result_components = (
                    scalar *
                    self.tensor._components[self.tensor._coordinates][self.tensor._indices]
                )
                result_metric = self.tensor._metric
                result_coordinates = self.tensor._coordinates
                result_indices = self.tensor._indices
                result_calc_indices = self.indices

        result_components = sym.Array(result_components) if not isinstance(result_components, sym.Array) else result_components

        result = Tensor(
            name="TEMP",
            components=result_components,
            metric=result_metric,
            coords=result_coordinates,
            indices=result_indices
        )
        return CalcObject(
            tensor=result,
            indices=result_calc_indices
        )

    def __rmul__(
        self: CalcObject,
        other: object
    ) -> CalcObject:
        """
        """
        return self * other

    def __truediv__(
        self: CalcObject,
        other: object
    ) -> CalcObject:
        """
        """

        # Make sure we are not multiplying a partial or covariant derivative
        if self.role != "tensor":
            raise ValueError(
                "Cannot divide a partial or covariant derivative."
            )

        # Make sure we are only multiplying by a scalar
        if not isinstance(other, (int, float, sym.Expr, sym.Symbol)):
            raise TypeError(
                f"Expected type '{int}', type '{float}', type '{sym.Symbol}', or type '{sym.Expr}' for argument 'other', got '{type(other)}'."
            )

        # Convert float to sympy objects
        other = sym.nsimplify(other)

        # Simply multiple all the components by the scalar
        result_components = (1/other) * self.tensor._components[self.tensor._coordinates][self.tensor._indices]   # type: ignore[operator]

        result = Tensor(
            name="TEMP",
            components=result_components,
            metric=self.tensor._metric,
            coords=self.tensor._coordinates,
            indices=self.tensor._indices
        )
        return CalcObject(
            tensor=result,
            indices=self.indices,
        )

    def __matmul__(
        self: CalcObject,
        other: object
    ) -> CalcObject:
        """
        """

        # Make sure we are only contracting with another calcObject
        if not isinstance(other, CalcObject):
            raise TypeError(
                f"Expected type '{CalcObject}' for argument 'other', got '{type(other)}'."
            )

        # Make sure both tensors are not partial or covariant derivatives
        if self.role in ["partial", "covariant"] and other.role in ["partial", "covariant"]:
            raise ValueError(
                "Cannot contract two partial or covariant derivatives."
            )

        # Make sure the metrics of both objects are the same
        if self.role == "tensor" and other.role == "tensor":
            if getattr(self.tensor, "_metric", None) != getattr(other.tensor, "_metric", None):
                raise ValueError(
                    "Cannot contract two objects with different Metric Tensors."
                )

        # Enumerate the indices of the first tensor and indices positions
        if not isinstance(self.indices, sym.Symbol):  # type: ignore[unreachable]
            symbols_self = {
                symbol: i
                for i, symbol in iter(enumerate(self.indices))
            }
        if isinstance(self.indices, sym.Symbol):  # type: ignore[unreachable]
            symbols_self = {
                self.indices: 0
            }

        # Enumerate the indices of the second tensor and indices positions
        if not isinstance(other.indices, sym.Symbol):  # type: ignore[unreachable]
            symbols_other = {
                symbol: i
                for i, symbol in iter(enumerate(other.indices))
            }
        if isinstance(other.indices, sym.Symbol):  # type: ignore[unreachable]
            symbols_other = {
                other.indices: 0
            }

        # Determine which indices are free
        free_indices = {
            symbol: (index, None) if symbol in symbols_self else (None, index)
            for symbol, index in iter(({**symbols_self, **symbols_other}).items())
            if symbol in symbols_self and symbol not in symbols_other
            or symbol in symbols_other and symbol not in symbols_self
        }

        # Determine which indices are to be contracted
        contracted_indices = {
            symbol: (symbols_self[symbol], symbols_other[symbol])
            for symbol in iter(({**symbols_self, **symbols_other}).keys())
            if symbol in symbols_self and symbol in symbols_other
        }

        result_indices_symbols = tuple(
            symbol
            for symbol in iter((*self.indices, *other.indices))
            if symbol in free_indices
        )

        # If both CalcObjects are tensors
        if self.role == "tensor" and other.role == "tensor":
            # If either of the objects is a scalar, just multiply them together
            if self.tensor._rank == 0 or other.tensor._rank == 0:
                return self * other

            # Retrieve the elements of each tensor
            components_self = self.tensor.get_components(
                coords=self.tensor._coordinates,
                indices=self.tensor._indices
            )

            other_indice_config = tuple(
                -self.tensor._indices[contracted_indices[symbol][0]]
                if symbol in contracted_indices
                else other.tensor._indices[i]
                for i, symbol in iter(enumerate(other.indices))
            ) if contracted_indices else other.tensor._indices

            components_other = other.tensor.get_components(
                coords=other.tensor._coordinates,
                indices=other_indice_config
            )

            result_components = contract(
                components_self,  # type: ignore[arg-type]
                components_other,  # type: ignore[arg-type]
                *tuple(
                    index_pair
                    for index_pair in iter(contracted_indices.values())
                )
            )

            # Configure result metric and coordinates
            result_indices = (
                *tuple(
                    index
                    for index, index_symbol in iter(zip(self.tensor._indices, self.indices))
                    if index_symbol in result_indices_symbols
                ),
                *tuple(
                    index
                    for index, index_symbol in iter(zip(other.tensor._indices, other.indices))
                    if index_symbol in result_indices_symbols
                )
            )
            result_metric = self.tensor._metric
            result_coordinates = self.tensor._coordinates

        # If one CalcObject is a partial derivative and the other is a tensor
        if self.role == "partial" and other.role == "tensor":
            # Determine contraction index and which indices are needed for the components
            components_indices = list(other.tensor._indices)
            if contracted_indices:
                components_indices[list(contracted_indices.items())[0][1][1]] = 1
            contraction_index = list(contracted_indices.items())[0][1][1] if contracted_indices else None

            # Retrieve the components of the tensor
            components = other.tensor.get_components(
                coords=other.tensor._coordinates,
                indices=tuple(components_indices)
            )

            # Perform the partial derivative calculation
            result_components = partial_contract(
                other.tensor._coordinates.get_components(),  # type: ignore[arg-type]
                components,  # type: ignore[arg-type]
                contraction_index
            )

            # Configure result metric and coordinates
            result_indices = tuple(
                index
                for index, index_symbol in iter(zip(other.tensor._indices, other.indices))
                if index_symbol in result_indices_symbols
            )
            result_indices = result_indices if contracted_indices else (-1, *result_indices)
            result_metric = other.tensor._metric
            result_coordinates = other.tensor._coordinates

        if self.role == "tensor" and other.role == "partial":
            # Determine contraction index and which indices are needed for the components
            components_indices = list(self.tensor._indices)
            if contracted_indices:
                components_indices[list(contracted_indices.items())[0][1][0]] = 1
            contraction_index = list(contracted_indices.items())[0][1][0] if contracted_indices else None

            # Retrieve the components of the tensor
            components = self.tensor.get_components(
                coords=self.tensor._coordinates,
                indices=tuple(components_indices)
            )

            # Perform the partial derivative calculation
            result_components = partial_contract(
                self.tensor._coordinates.get_components(),  # type: ignore[arg-type]
                components,  # type: ignore[arg-type]
                contraction_index
            )

            # Configure result metric and coordinates
            result_indices = tuple(
                index
                for index, index_symbol in iter(zip(self.tensor._indices, self.indices))
                if index_symbol in result_indices_symbols
            )
            result_indices = result_indices if contracted_indices else (-1, *result_indices)
            result_metric = self.tensor._metric
            result_coordinates = self.tensor._coordinates

        # If one CalcObject is a covariant derivative and the other is a tensor
        if self.role == "covariant" and other.role == "tensor":
            # Result indice configuration
            calc_indices = " ".join([str(symbol) for symbol in result_indices_symbols])

            # Retrieve the appropriate Christoffel symbols
            christoffel = CalcChristoffel(other.tensor._metric)

            # Create two dummy symbol used for contractions
            dummy_symbols: List[sym.Symbol] = []
            for symbol in options.ALL_SYMBOLS:
                if len(dummy_symbols) != 2:
                    if symbol not in [*other.indices, *self.indices]:
                        dummy_symbols.append(symbol)
            dummy_symbol = dummy_symbols[0]
            partial_symbol = self.indices[0] if other.indices and self.indices[0] not in other.indices else dummy_symbols[1]

            # Determine the index configuration of the tensor's indices
            tensor_configs = tuple(
                " ".join(
                    [
                        str(index) if count != j else str(dummy_symbol)
                        for count, index in enumerate(other.indices)
                    ]
                )
                for j in range(len(other.indices))
            )

            # Determine upper indices
            upper = tuple(
                symbol
                for symbol, index in zip(other.indices, other.tensor._indices)
                if index == 1
            )
            tensor_upper_configs = tuple(
                config
                for config, index in zip(tensor_configs, other.tensor._indices)
                if index == 1
            )
            christoffel_upper_configs = tuple(
                " ".join([str(symbol), str(partial_symbol), str(dummy_symbol)])
                for symbol in upper
            )

            # Determine lower indices
            lower = tuple(
                symbol
                for symbol, index in zip(other.indices, other.tensor._indices)
                if index == -1
            )
            tensor_lower_configs = tuple(
                config
                for config, index in zip(tensor_configs, other.tensor._indices)
                if index == -1
            )
            christoffel_lower_configs = tuple(
                " ".join([str(dummy_symbol), str(partial_symbol), str(symbol)])
                for symbol in lower
            )

            # Define a temporary function that we can map over
            def mat_mul_other(config1: str, config2: str) -> CalcObject:
                if christoffel is None or other.tensor is None:  # type: ignore[attr-defined]
                    raise ValueError("Christoffel symbols and/or tensor are not defined")
                return christoffel(config1) @ other.tensor(config2)  # type: ignore[attr-defined, no-any-return]

            # Perform the calculation
            calc_object = PartialD(str(partial_symbol)) @ other
            calc_object += sum(
                list(
                    map(
                        mat_mul_other,
                        christoffel_upper_configs,
                        tensor_upper_configs
                    )
                )
            )
            calc_object -= sum(
                list(
                    map(
                        mat_mul_other,
                        christoffel_lower_configs,
                        tensor_lower_configs
                    )
                )
            )

            if calc_object.tensor is None:
                raise ValueError("CalcObject does not have a tensor, unknown error.")

            result = Calc(
                calc_object.tensor(" ".join([str(symbol) for symbol in [*self.indices, *other.indices]])),
                indices=calc_indices
            )
            result_components = result.get_components()  # type: ignore[assignment]
            result_components = sym.simplify(result_components) if result_components.rank() else sym.Array(sym.simplify(result_components[()]))

            # Configure result metric and coordinates
            result_indices = tuple(
                index
                for index, index_symbol in iter(zip(other.tensor._indices, other.indices))
                if index_symbol in result_indices_symbols
            )
            result_indices = result_indices if contracted_indices else (-1, *result_indices)
            result_metric = other.tensor._metric
            result_coordinates = other.tensor._coordinates

            # Delete the temporary result tensor
            result.delete(silent=True)

        if self.role == "tensor" and other.role == "covariant":
            # Result indice configuration
            calc_indices = " ".join([str(symbol) for symbol in result_indices_symbols])

            # Retrieve the appropriate Christoffel symbols
            christoffel = CalcChristoffel(self.tensor._metric)

            # Create two dummy symbol used for contractions
            dummy_symbols = []
            for symbol in options.ALL_SYMBOLS:
                if len(dummy_symbols) != 2:
                    if symbol not in [*self.indices, *other.indices]:
                        dummy_symbols.append(symbol)
            dummy_symbol = dummy_symbols[0]
            partial_symbol = other.indices[0] if other.indices[0] not in other.indices else dummy_symbols[1]

            # Determine the index configuration of the tensor's indices
            tensor_configs = tuple(
                " ".join(
                    [
                        str(index) if count != j else str(dummy_symbol)
                        for count, index in enumerate(self.indices)
                    ]
                )
                for j in range(len(self.indices))
            )

            # Determine upper indices
            upper = tuple(
                symbol
                for symbol, index in zip(self.indices, self.tensor._indices)
                if index == 1
            )
            tensor_upper_configs = tuple(
                config
                for config, index in zip(tensor_configs, self.tensor._indices)
                if index == 1
            )
            christoffel_upper_configs = tuple(
                " ".join([str(symbol), str(partial_symbol), str(dummy_symbol)])
                for symbol in upper
            )

            # Determine lower indices
            lower = tuple(
                symbol
                for symbol, index in zip(self.indices, self.tensor._indices)
                if index == -1
            )
            tensor_lower_configs = tuple(
                config
                for config, index in zip(tensor_configs, self.tensor._indices)
                if index == -1
            )
            christoffel_lower_configs = tuple(
                " ".join([str(dummy_symbol), str(partial_symbol), str(symbol)])
                for symbol in lower
            )

            # Define a temporary function that we can map over
            def mat_mul_self(config1: str, config2: str) -> CalcObject:
                if christoffel is None or self.tensor is None:
                    raise ValueError("Christoffel symbols and/or tensor are not defined")
                return christoffel(config1) @ self.tensor(config2)

            # Perform the calculation
            calc_object = PartialD(str(partial_symbol)) @ self
            calc_object += sum(
                list(
                    map(
                        mat_mul_self,
                        christoffel_upper_configs,
                        tensor_upper_configs
                    )
                )
            )
            calc_object -= sum(
                list(
                    map(
                        mat_mul_self,
                        christoffel_lower_configs,
                        tensor_lower_configs
                    )
                )
            )

            if calc_object.tensor is None:
                raise ValueError("CalcObject does not have a tensor, unknown error.")

            result = Calc(
                calc_object.tensor(" ".join([str(symbol) for symbol in [*self.indices, *other.indices]])),
                indices=calc_indices
            )
            result_components = result.get_components()  # type: ignore[assignment]
            result_components = sym.simplify(result_components) if result_components.rank() else sym.Array(sym.simplify(result_components[()]))

            # Configure result metric and coordinates
            result_indices = tuple(
                index
                for index, index_symbol in iter(zip(self.tensor._indices, self.indices))
                if index_symbol in result_indices_symbols
            )
            result_indices = result_indices if contracted_indices else (-1, *result_indices)
            result_metric = self.tensor._metric
            result_coordinates = self.tensor._coordinates

            # Delete the temporary result tensor
            result.delete(silent=True)

        # Create a new tensor, and return the corresponding CalcObject
        result_tensor = Tensor(
            name="TEMP",
            components=result_components if isinstance(result_components, sym.Array) else sym.Array(result_components),
            metric=result_metric,
            coords=result_coordinates,
            indices=result_indices
        )

        return CalcObject(
            tensor=result_tensor,
            indices=result_indices_symbols,
        )


def PartialD(
    index: Union[str, sym.Symbol] = ""
) -> CalcObject:
    """
    (function) PartialD(index)

    Represents the partial derivative when used within Calc()

    `index`: The index of the partial derivative used in tensor formulae.
    """

    if isinstance(index, str) and index != "":
        index = sym.symbols(index, seq=True)
        if len(index) != 1:  # type: ignore[arg-type]
            raise ValueError(
                "Cannot create a partial derivative with more than one index."
            )
    return CalcObject(
        tensor=None,
        indices=index,  # type: ignore[arg-type]
        role="partial"
    )


def CovariantD(
    index: Union[str, sym.Symbol] = ""
) -> CalcObject:
    """
    (function) CovariantD(index)

    Represents the covariant derivative when used within Calc()

    `index`: The index of the covariant derivative used in tensor formulae.
    """
    if isinstance(index, str) and index != "":
        index = sym.symbols(index, seq=True)
        if len(index) != 1:  # type: ignore[arg-type]
            raise ValueError(
                "Cannot create a partial derivative with more than one index."
            )
    return CalcObject(
        tensor=None,
        indices=index,  # type: ignore[arg-type]
        role="covariant"
    )


def Calc(
    calc_object: CalcObject,
    name: str = "Result",
    symbol: Optional[Union[str, sym.Symbol]] = "",
    indices: Optional[str] = None
) -> Tensor:
    """
    (function) Calc(calc_object, name, symbol, indices)

    Calculates a tensor formula.

    `calc_object`: The tensor formula.
    `name`: Defines the name of the resulting tensor. Defaults to "Result".
    `symbol`: Defines the symbol used to represent the resulting tensor. A placeholder symbol will be used by default.
    `indices`: A string representing the order of indices of the resulting tensor. Defaults to the order that appears in the tensor formula.
    """

    indices = None if not indices else indices

    # Make sure we are constructing the result from a CalcObject
    if not isinstance(calc_object, CalcObject):
        raise TypeError(
            f"Expected type '{CalcObject}' for argument 'calc_object', got '{type(calc_object)}'."
        )

    # If result indices are supplied, ensure they are valid
    if indices is not None:
        if not isinstance(indices, str):
            raise TypeError(
                f"Expected type '{str}' for argument 'indices', got '{type(indices)}'."
            )
        indices = sym.symbols(indices, seq=True)

        if indices not in list(permutations(calc_object.indices)):  # type: ignore[comparison-overlap]
            raise ValueError(
                f"Result indices {indices} must be the same up to a permutation of the input indices {calc_object.indices}."
            )

    # Otherwise, use the indices of the CalcObject
    if indices is None:
        indices = calc_object.indices  # type: ignore[assignment]

    # Enumerate the indices of the CalcObject
    result_indices = {
        symbol: i
        for i, symbol in enumerate(iter(calc_object.indices))
    }

    # Enumerate the indices of the requested result
    requested_indices = {
        symbol: i
        for i, symbol in enumerate(iter(indices))
    }
    permutation = tuple(
        result_indices[symbol]
        for symbol in requested_indices
    )
    calc_object_indices = tuple(
        calc_object.tensor._indices[i]
        for i in permutation
    )

    # Retrieve the components of the CalcObject
    calc_object_components = calc_object.tensor.get_components(
        coords=calc_object.tensor._coordinates,
        indices=calc_object.tensor._indices
    )
    calc_object_components = sym.permutedims(calc_object_components, permutation)

    # Simplify the components
    if calc_object_components.rank() > 0:
        calc_object_components = sym.simplify(calc_object_components)

    # Set result tensor indices and symbol
    kwargs = {}
    kwargs["symbol"] = symbol if symbol is not None else ""

    result = Tensor(
        name=name,
        components=calc_object_components,
        metric=calc_object.tensor._metric,
        coords=calc_object.tensor._coordinates,
        indices=calc_object_indices,
        **kwargs
    )

    for _ in range(2):
        for instance in calc_object.tensor._instances:
            if instance._name == "TEMP":
                instance.delete(silent=True)

    return result


def CalcChristoffel(
    metric: Metric,
) -> Christoffel:
    """
    Calculates the Christoffel Symbols (the coefficients of the Levi-Cevita connection)
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    # If the Christoffel Symbols have already been calculated, return them
    for instance in metric._instances:
        if instance._name == metric._name+" Christoffel":
            return instance  # type: ignore[return-value]

    # Create a new temporary inverse metric to avoid having to raise the first index later
    inverse_metric = copy.copy(metric).change_default_indices(indices=(1, 1))

    inverse_metric._name = "TEMP"

    # Calculate the Christoffel Symbols
    christoffel = Calc(
        sym.Rational(1, 2) * inverse_metric("lambda sigma") @
        (PartialD("mu") @ metric("nu sigma") + PartialD("nu") @ metric("sigma mu") - PartialD("sigma") @ metric("mu nu"))
    )

    # Get the components with the correct index configuration
    components = christoffel.get_components(indices=(1, -1, -1))

    result = Christoffel(
        name=metric._name+" Christoffel",
        components=components,
        metric=metric,
        coords=metric._coordinates,
        indices=(1, -1, -1),
        symbol=sym.Symbol("Gamma")
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    christoffel.delete(silent=True)

    return result


def CalcRiemannTensor(
    metric: Metric,
) -> Tensor:
    """
    Calculates the Riemann Tensor
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    # If the Riemann Tensor has already been calculated, return it
    for instance in metric._instances:
        if instance._name == metric._name+" Riemann":
            return instance  # type: ignore[return-value]

    # Calculate the Christoffel Symbols
    christoffel = CalcChristoffel(metric)

    # Calculate the Riemann Tensor
    riemann_tensor = Calc(
        PartialD("mu") @ christoffel("rho nu sigma")
        - PartialD("nu") @ christoffel("rho mu sigma")
        + christoffel("rho mu lambda") @ christoffel("lambda nu sigma")
        - christoffel("rho nu lambda") @ christoffel("lambda mu sigma"),
        indices="rho sigma mu nu"
    )

    # Get the components with the correct index configuration
    components = riemann_tensor.get_components(indices=(1, -1, -1, -1))

    result = Tensor(
        name=metric._name+" Riemann",
        components=components,
        metric=metric,
        coords=metric._coordinates,
        indices=(1, -1, -1, -1),
        symbol=sym.Symbol("R")
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    riemann_tensor.delete(silent=True)

    return result


def CalcRicciTensor(
    metric: Metric,
) -> Tensor:
    """
    Calculates the Ricci Tensor
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    # If the Ricci Tensor has already been calculated, return it
    for instance in metric._instances:
        if instance._name == metric._name+" Ricci Tensor":
            return instance  # type: ignore[return-value]

    # Calculate the Riemann tensor
    riemann_tensor = CalcRiemannTensor(metric)

    # Calculate the Ricci Tensor
    ricci_tensor = Calc(
        riemann_tensor("lambda mu lambda nu")
    )

    # Get the components with the correct index configuration
    components = ricci_tensor.get_components(indices=(-1, -1))

    result = Tensor(
        name=metric._name+" Ricci Tensor",
        components=components,
        metric=metric,
        coords=metric._coordinates,
        indices=(-1, -1),
        symbol=sym.Symbol("R")
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    ricci_tensor.delete(silent=True)

    return result


def CalcWeylTensor(
    metric: Metric,
) -> Tensor:
    """
    Calculates the Weyl Tensor
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    # If the Weyl Tensor has already been calculated, return it
    for instance in metric._instances:
        if instance._name == metric._name+" Weyl Tensor":
            return instance  # type: ignore[return-value]

    # Determine the number of dimensions
    n = len(metric._coordinates.get_components())

    # Calculate the Riemann tensor
    riemann_tensor = CalcRiemannTensor(metric)

    # Calculate the Ricci Tensor
    ricci_tensor = CalcRicciTensor(metric)

    # Calculate the Ricci Scalar
    ricci_scalar = CalcRicciScalar(metric)

    # Calculate the Weyl Tensor
    weyl_tensor = Calc(
        riemann_tensor("i k l m") + (1/(n-2)) * (
            ricci_tensor("i m") @ metric("k l") -
            ricci_tensor("i l") @ metric("k m") +
            ricci_tensor("k l") @ metric("i m") -
            ricci_tensor("k m") @ metric("i l")
        ) +
        (1/((n-2)*(n-1))) * (
            ricci_scalar() @ (metric("i l") @ metric("k m") - metric("i m") @ metric("k l"))
        ),
        indices="i k l m"
    )

    # Get the components with the correct index configuration
    components = weyl_tensor.get_components(indices=(-1, -1, -1, -1))

    result = Tensor(
        name=metric._name+" Weyl Tensor",
        components=components,
        metric=metric,
        coords=metric._coordinates,
        indices=(-1, -1, -1, -1),
        symbol=sym.Symbol("C")
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    ricci_tensor.delete(silent=True)

    return result


def CalcRicciScalar(
    metric: Metric,
) -> Tensor:
    """
    Calculates the Ricci Scalar
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    # If the Ricci Scalar has already been calculated, return it
    for instance in metric._instances:
        if instance._name == metric._name+" Ricci Scalar":
            return instance  # type: ignore[return-value]

    # Calculate the Ricci tensor
    ricci_tensor = CalcRicciTensor(metric)

    # Calculate the Ricci scalar
    ricci_scalar = Calc(
        ricci_tensor("lambda lambda")
    )

    # Get the components with the correct index configuration
    components = ricci_scalar.get_components()

    result = Tensor(
        name=metric._name+" Ricci Scalar",
        components=components,
        metric=metric,
        coords=metric._coordinates,
        indices=(),
        symbol=sym.Symbol("R")
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    ricci_scalar.delete(silent=True)

    return result


def CalcEinsteinTensor(
    metric: Metric,
) -> Tensor:
    """
    Calculates the Einstein Tensor
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    # If the Einstein Tensor has already been calculated, return it
    for instance in metric._instances:
        if instance._name == metric._name+" Einstein":
            return instance  # type: ignore[return-value]

    # Calculate the Ricci tensor and Ricci scalar
    ricci_tensor = CalcRicciTensor(metric)
    ricci_scalar = CalcRicciScalar(metric)

    # Calculate the Einstein Tensor
    einstein = Calc(
        ricci_tensor("mu nu") - sym.Rational(1, 2) * ricci_scalar("") @ metric("mu nu")
    )

    # Get the components with the correct index configuration
    components = einstein.get_components(indices=(-1, -1))

    result = Tensor(
        name=metric._name+" Einstein",
        components=components,
        metric=metric,
        coords=metric._coordinates,
        indices=(-1, -1),
        symbol=sym.Symbol("G")
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    einstein.delete(silent=True)

    return result


def CalcNormSquared(
    tensor: BaseTensor,
) -> Tensor:
    """
    Calculates the Einstein Tensor
    """

    if not isinstance(tensor, (Tensor, Metric)):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(tensor)}'."
        )

    # If the Einstein Tensor has already been calculated, return it
    for instance in tensor._instances:
        if instance._name == tensor._name+" Norm Squared":
            return instance  # type: ignore[return-value]

    # Create dummy indices to contract over
    symbols = sym.symbols(
        f"x:{len(tensor._indices)}",
        seq=True
    )
    symbols = " ".join(
        [
            str(symbol)
            for symbol in symbols
        ]
    )

    # Calculate the Norm Squared
    norm_squared = Calc(
        tensor(symbols) @ tensor(symbols)
    )

    # Get the components with the correct index configuration
    components = norm_squared.get_components()

    result = Tensor(
        name=tensor._name+" Norm Squared",
        components=components,
        metric=tensor._metric,
        coords=tensor._coordinates,
        indices=(),
        symbol=""
    )

    # Delete the temporary tensors
    for instance in tensor._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    norm_squared.delete(silent=True)

    return result


def CalcLagrangian(
    metric: Metric,
    coords: Optional[Coordinates] = None
) -> Lagrangian:
    """
    Calculates the Lagrangian.
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    if coords is not None and not isinstance(coords, Coordinates):
        raise TypeError(
            f"Expected type '{Coordinates}' for argument 'coords', got '{type(coords)}'."
        )

    coords = coords if coords is not None else metric._coordinates

    # If the Lagrangian has already been calculated, return it
    for instance in metric._instances:
        if isinstance(instance, Lagrangian) and instance._name == metric._name+" Lagrangian":
            if instance._coordinates == coords:
                return instance
            return instance.change_default_coords(coords)  # type: ignore[return-value]

    coordinates = coords.get_components()

    # Create a new tangent vector
    v_components = sym.Array(
        [
            sym.Symbol(
                str(coord)+"dot"
            )
            for coord in coordinates
        ]
    )
    temp_v_vector = metric.new_tensor(
        name="TEMP",
        components=v_components,
        coords=coords,
        indices=(1,)
    )

    # Calculate the Lagrangian
    lagrangian = Calc(
        temp_v_vector("mu") @ temp_v_vector("mu")
    )

    # Get the components with the correct index configuration
    components = lagrangian.get_components()

    result = Lagrangian(
        name=metric._name+" Lagrangian",
        components=components,
        metric=metric,
        coords=coords,
        indices=(),
        symbol=sym.Symbol("L")
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    lagrangian.delete(silent=True)

    return result


def CalcGeodesicFromLagrangian(
    metric: Metric,
    coords: Optional[Coordinates] = None,
    activate: bool = True
) -> GeodesicLagrangian:
    """
    Calculates the Geodesic equations from the Lagrangian.
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    if coords is not None and not isinstance(coords, Coordinates):
        raise TypeError(
            f"Expected type '{Coordinates}' for argument 'coords', got '{type(coords)}'."
        )

    coords = coords if coords is not None else metric._coordinates

    # If the Geodesics have already been calculated, return it
    for instance in metric._instances:
        if isinstance(instance, GeodesicLagrangian) and instance._name == metric._name+" Geodesic From Lagrangian":
            if instance._coordinates == coords:
                return instance
            return instance.change_default_coords(coords)  # type: ignore[return-value]

    parameter = options.CURVE_PARAMETER

    lagrangian_components = sym.Rational(1, 2) * metric.calc_lagrangian(coords).get_components(coords=coords)[()]  # type: ignore[index]

    coordinates = coords.get_components()
    coordinates_dot = sym.Array(
        [
            sym.symbols(str(coord)+"dot")
            for coord in coordinates
        ]
    )
    coordinates_ddot = sym.Array(
        [
            sym.symbols(str(coord)+"ddot")
            for coord in coordinates
        ]
    )
    f_coordinates = sym.Array(
        [
            sym.Function(str(coord))(parameter)
            for coord in coordinates
        ]
    )
    f_coordinates_dot = sym.Array(
        [
            sym.Function(str(coord) + "dot")(parameter)
            for coord in coordinates
        ]
    )

    dl_dx = dl_dx = lagrangian_components.diff(f_coordinates)

    dl_dxdot = lagrangian_components.diff(f_coordinates_dot)

    if activate:
        d_dl_dxdot = sym.Array(
            [
                component.diff(parameter, evaluate=True)
                for component in dl_dxdot
            ]
        )

    if not activate:
        d_dl_dxdot = sym.Array(
            [
                component.diff(parameter, evaluate=False)
                for component in dl_dxdot
            ]
        )

    components = d_dl_dxdot - dl_dx

    components = components.subs(
        {
            **{
                f_coord: coord
                for f_coord, coord in zip(f_coordinates, coordinates)
            },
            **{
                f_coord_dot: coord_dot
                for f_coord_dot, coord_dot in zip(f_coordinates_dot, coordinates_dot)
            },
            **{
                f_coord.diff(parameter): coord_dot
                for f_coord, coord_dot in zip(f_coordinates, coordinates_dot)
            },
            **{
                f_coord_dot.diff(parameter): coord_ddot
                for f_coord_dot, coord_ddot in zip(f_coordinates_dot, coordinates_ddot)
            }
        }
    )

    # Simplify the components by creating an expression of the form 0 = p/q, and multiplying by q
    if activate:
        components = components.as_mutable()
        for i, component in enumerate(components):
            if isinstance(component, sym.Expr):
                component = sym.cancel(component)
                if component.func == sym.Mul and len(component.args) == 2:
                    args1, args2 = component.args
                    if args1.func == sym.Pow and sym.Integer(-1) == args1.args[1]:
                        component = args2
                    if args2.func == sym.Pow and sym.Integer(-1) == args2.args[1]:
                        component = args1
                components[(i,)] = sym.simplify(component)
        components = components.as_immutable()

    result = GeodesicLagrangian(
        name=metric._name+" Geodesic From Lagrangian",
        components=components,
        coords=coords,
        metric=metric,
        indices=(1,)
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)

    return result


def CalcGeodesicFromChristoffel(
    metric: Metric,
    coords: Optional[Coordinates] = None
) -> GeodesicChristoffel:
    """
    Calculates the Geodesic equations from the Christoffel symbols.
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    if coords is not None and not isinstance(coords, Coordinates):
        raise TypeError(
            f"Expected type '{Coordinates}' for argument 'coords', got '{type(coords)}'."
        )

    coords = coords if coords is not None else metric._coordinates

    # If the Geodesics have already been calculated, return it
    for instance in metric._instances:
        if isinstance(instance, GeodesicChristoffel) and instance._name == metric._name+" Geodesic From Christoffel":
            if instance._coordinates == coords:
                return instance
            return instance.change_default_coords(coords)  # type: ignore[return-value]

    # Retrieve the coordinates
    coordinates = coords.get_components()

    # Calculate the Christoffel symbols
    christoffel = metric.calc_christoffel()

    # Create a new tangent vector
    v_components = sym.Array(
        [
            sym.Symbol(
                str(coord)+"dot"
            )
            for coord in coordinates
        ]
    )
    temp_v_vector = metric.new_tensor(
        name="TEMP",
        components=v_components,
        coords=coords,
        indices=(1,)
    )

    a_components = sym.Array(
        [
            sym.Symbol(
                str(coord)+"ddot"
            )
            for coord in coordinates
        ]
    )
    temp_a_vector = metric.new_tensor(
        name="TEMP",
        components=a_components,
        coords=coords,
        indices=(1,)
    )

    # Calculate the components of the geodesic equations
    geodesic = Calc(
        temp_a_vector("sigma") + christoffel("sigma mu nu") @ temp_v_vector("mu") @ temp_v_vector("nu")
    )

    components = geodesic.get_components()

    # Simplify the components by creating an expression of the form 0 = p/q, and multiplying by q
    components = components.as_mutable()  # type: ignore[union-attr]
    for i, component in enumerate(components):
        if isinstance(component, sym.Expr):
            component = sym.cancel(component)
            if component.func == sym.Mul and len(component.args) == 2:
                args1, args2 = component.args
                if args1.func == sym.Pow and sym.Integer(-1) == args1.args[1]:
                    component = args2
                if args2.func == sym.Pow and sym.Integer(-1) == args2.args[1]:
                    component = args1
            components[(i,)] = sym.simplify(component)  # type: ignore[index]
    components = components.as_immutable()  # type: ignore[union-attr]

    result = GeodesicChristoffel(
        name=metric._name+" Geodesic From Christoffel",
        components=components,
        coords=coords,
        metric=metric,
        indices=(1,)
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    geodesic.delete(silent=True)

    return result


def CalcGeodesicWithTimeParameter(
    metric: Metric,
    coords: Optional[Coordinates] = None
) -> GeodesicTime:
    """
    Calculates the Geodesic equations from the Christoffel symbols, with respect to the time coordinate.
    """

    if not isinstance(metric, Metric):
        raise TypeError(
            f"Expected type '{Metric}' for argument 'metric', got '{type(metric)}'."
        )

    if coords is not None and not isinstance(coords, Coordinates):
        raise TypeError(
            f"Expected type '{Coordinates}' for argument 'coords', got '{type(coords)}'."
        )

    coords = coords if coords is not None else metric._coordinates

    # If the Geodesics have already been calculated, return it
    for instance in metric._instances:
        if isinstance(instance, GeodesicTime) and instance._name == metric._name+" Geodesic With Time Parameter":
            if instance._coordinates == coords:
                return instance
            return instance.change_default_coords(coords)  # type: ignore[return-value]

    # Retrieve the coordinates
    coordinates = coords.get_components()

    # Calculate the Christoffel symbols
    christoffel = metric.calc_christoffel()

    # Create a new tangent vector
    v_components = sym.Array(
        [
            sym.Symbol(
                str(coord)+"dot"
            )
            if i != 0 else 1
            for i, coord in enumerate(coordinates)
        ]
    )
    temp_v_vector = metric.new_tensor(
        name="TEMP",
        components=v_components,
        coords=coords,
        indices=(1,)
    )

    a_components = sym.Array(
        [
            sym.Symbol(
                str(coord)+"ddot"
            )
            if i != 0 else 0
            for i, coord in enumerate(coordinates)
        ]
    )
    temp_a_vector = metric.new_tensor(
        name="TEMP",
        components=a_components,
        coords=coords,
        indices=(1,)
    )

    # We need the christoffel symbols with index configuration "0 mu nu"
    christoffel0_components = christoffel.get_components(coords=coords)[(0, slice(0, None), slice(0, None))]  # type: ignore[index]
    christoffel0 = metric.new_tensor(
        name="TEMP",
        components=christoffel0_components,
        coords=coords,
        indices=(-1, -1)
    )

    # Calculate the components of the geodesic equations
    geodesic = Calc(
        temp_a_vector("sigma") +
        (christoffel("sigma mu nu") - christoffel0("mu nu") @ temp_v_vector("sigma")) @
        temp_v_vector("mu") @ temp_v_vector("nu")
    )

    components = geodesic.get_components()

    # Simplify the components by creating an expression of the form 0 = p/q, and multiplying by q
    components = components.as_mutable()  # type: ignore[union-attr]
    for i, component in enumerate(components):
        if isinstance(component, sym.Expr):
            component = sym.cancel(component)
            if component.func == sym.Mul and len(component.args) == 2:
                args1, args2 = component.args
                if args1.func == sym.Pow and sym.Integer(-1) == args1.args[1]:
                    component = args2
                if args2.func == sym.Pow and sym.Integer(-1) == args2.args[1]:
                    component = args1
            components[(i,)] = sym.simplify(component)  # type: ignore[index]
    components = components.as_immutable()  # type: ignore[union-attr]

    result = GeodesicTime(
        name=metric._name+" Geodesic With Time Parameter",
        components=components,
        coords=coords,
        metric=metric,
        indices=(1,)
    )

    # Delete the temporary tensors
    for instance in metric._instances:
        if instance._name == "TEMP":
            instance.delete(silent=True)
    geodesic.delete(silent=True)

    return result
