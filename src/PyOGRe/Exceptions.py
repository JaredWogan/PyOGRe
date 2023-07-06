from __future__ import annotations

from typing import Tuple, Union


# Indices Dimensions Error, raised if supplied indices have invalid dimensions
# error = "tensor", "metric"
class IndicesDimensionError(Exception):
    """
    Error class used when supplied indices are of invalid dimensions.
    """

    def __init__(self: IndicesDimensionError, indices: Union[str, Tuple[int, ...]], error: str, dim: int) -> None:
        self.message = f"The supplied indices ({indices}) are invalid."
        self.extra = ""
        if error == "tensor":
            self.extra = f"\nThe number of indices ({len(indices)}) must match the rank of the Tensor ({dim})."
        if error == "metric":
            self.extra = f"\nThe number of indices ({len(indices)}) for a Metric must be {dim}."
        super().__init__(self.message)

    def __str__(self: IndicesDimensionError) -> str:
        return self.message + self.extra


# Indices Error, raised if supplied indices are invalid
# error = "tensor", "coordinates"
class IndicesValueError(Exception):
    """
    Error class used when supplied indices contain invalid values.
    """

    def __init__(self: IndicesValueError, indices: Tuple[int, ...], error: str = "tensor") -> None:
        self.message = f"The supplied indices {indices} are invalid."
        self.extra = ""
        if error == "tensor":
            self.extra = "\nAll indices must either be +1 or -1."
        if error == "coordinates":
            self.extra = "\nCoordinates may only have one indice, which must have value +1."
        super().__init__(self.message)

    def __str__(self: IndicesValueError) -> str:
        return self.message + self.extra


# Tensor Error, raised if supplied Tensor is incompatible with the supplied Coordinates
class TensorDimensionError(Exception):
    """
    Error class used when supplied tensor is of invalid dimensions.
    """

    def __init__(self: TensorDimensionError, tensor_shape: Tuple[int, ...]) -> None:
        self.message = f"Tensor components must have uniform shape, got {tensor_shape}, which is not uniform."
        super().__init__(self.message)

    def __str__(self: TensorDimensionError) -> str:
        return self.message


# Tensor Error, raised if supplied Tensor is incompatible with the supplied Coordinates
class TensorCoordinatesError(Exception):
    """
    Error class used when supplied indices contain invalid values.
    """

    def __init__(self: TensorCoordinatesError, tensor_shape: Tuple[int, ...], coordinates_shape: Tuple[int, ...]) -> None:
        self.message = f"Tensor components must have the same dimensions {tensor_shape} as the coordinates {coordinates_shape}."
        super().__init__(self.message)

    def __str__(self: TensorCoordinatesError) -> str:
        return self.message


# Transformation Error, raised if transformation rules are not present or a transformation is not valid
# error = "coords", "dimensions", "rules", "transformation"
class TransformationError(Exception):
    """
    Error class used when supplied transformations are invalid.
    """

    def __init__(self: TransformationError, error: str, info: Union[str, Tuple[str, ...], Tuple[int, ...]]) -> None:
        self.message = "The transformation is invalid."
        self.extra = ""
        if error == "type":
            self.extra = f"\nExpected type 'Coordinates' for argument 'coords', got '{info}'."
        if error == "coords":
            self.extra = "\nThe coords cannot be the same Coordinate system."
        if error == "dimensions":
            self.extra = f"\nThe dimensions must be the same for the coords ({info[0]}) and the transformation ({info[1]})."
        if error == "rules":
            self.extra = f"\nNo coordinate transformation rule exists between {info[0]} and {info[1]}."
        if error == "transformation":
            self.extra = f"\nCannot transform the Tensor {info[0]} from the Coordinate system {info[1]} to the Coordinate system {info[2]} with indice configuration {info[3]}."
        super().__init__(self.message)

    def __str__(self: TransformationError) -> str:
        return self.message + self.extra
