from __future__ import annotations

from typing import List, Optional, Tuple, Union

import sympy as sym

from PyOGRe.Coordinates import Coordinates
from PyOGRe.Metric import Metric
from PyOGRe.Tensor import Tensor

__doc__ = """
Christoffel Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


class Christoffel(Tensor):
    """
    Christoffel Class. Derived from Tensor, simply adds specialized tranformation methods.
    """

    def __init__(
        self: Christoffel,
        name: str,
        components: Union[sym.Array, sym.Matrix],  # type: ignore[valid-type]
        metric: Metric,
        coords: Coordinates,
        indices: Optional[Tuple[int, int, int]] = (1, -1, -1),
        symbol: Union[str, sym.Symbol] = sym.Symbol("Gamma")
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
        self: Christoffel,
        coords: Coordinates,
        indices: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Specialized transform_coordinates method for Christoffel 'Tensors'.
        """
        if indices is None:
            indices = self._indices

        # Check to see if the Jacobians for the coords Coordinates have already been calculated
        self._coordinates._check_jacobians(coords=coords)

        # Retrieve the coordinates
        self_coordinates = self._coordinates.get_components()

        # Retrieve the needed Jacobian, Inverse Jacobian, Christoffel Jacobian, the transformation rules, and the required components
        components = self._components[self._coordinates][indices]
        jacobian = self._coordinates._jacobians[coords]
        inverse_jacobian = self._coordinates._inverse_jacobians[coords]
        christoffel_jacobian = self._coordinates._christoffel_jacobians[coords]
        rules = self._coordinates._transformations[coords]

        # Create an ordered list of rules
        ordered_rules: List[sym.Eq] = []
        for coord in self_coordinates:
            exists = False
            for rule in rules:
                if rule is not None and rule.lhs == coord:
                    exists = True
                    ordered_rules.append(rule)
            if not exists:
                ordered_rules.append(None)

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

        # Next, we need to add the additional term, which means the Chirstoffel symbols don't transform like a tensor
        additional_term = sym.tensorproduct(
            inverse_jacobian,
            christoffel_jacobian
        )
        additional_term = sym.tensorcontraction(
            additional_term,
            (1, 2)
        )
        transformed_tensor = transformed_tensor + additional_term

        # Once computed, go through and substitute the transformation rules
        # In order to simplify the expression in the new Coordinate system
        for i, rule in enumerate(ordered_rules):
            if rule is not None:
                symbol = self._coordinates._components[self._coordinates][(1,)][i]
                substitute = sym.solve(rule, self._coordinates._components[self._coordinates][(1,)][i])[0]
                transformed_tensor = transformed_tensor.subs({symbol: substitute})

        # Check if the destination coordinates are already in the dictionary
        # If not, create a new dictionary for the coords coordinates
        if coords not in self._components:
            self._components[coords] = {}

        # Store the transformed tensor in the dictionary after simplifying
        transformed_tensor = sym.simplify(transformed_tensor)
        self._components[coords][indices] = transformed_tensor
