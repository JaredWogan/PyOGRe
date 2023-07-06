from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, Optional, Tuple

import sympy as sym

__doc__ = """
Utils Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


def str_symbols(
    names: str
) -> str:
    """
    Creates a string of symbols, which can be used inside the Calc function.

    Passes the argument `names` to the SymPy symbols function, the returns a string of the resulting symbols.

    Example:

    >>> from PyOGRe import str_symbols
    >>> str_symbols("x0:10")

    This will return a string "x0 x1 x2 x3 x4 x5 x6 x7 x8 x9".
    """

    if not isinstance(names, str):
        raise TypeError("Argument 'names' must be string.")

    return " ".join(
        str(symbol) for symbol in sym.symbols(names, seq=True)
    )


def zero_tensor(
    dim: int,
    rank: int
) -> sym.MutableDenseNDimArray:
    """
    Create a zero tensor of the specified dimension and rank.
    """

    if not isinstance(dim, (int, sym.Integer)):
        raise TypeError("Argument 'dim' must be an integer.")
    if not isinstance(rank, (int, sym.Integer)):
        raise TypeError("Argument 'rank' must be an integer.")

    if rank == 0:
        return sym.MutableDenseNDimArray(0)

    # If we aren't returning a scalar, make sure dim is a positive integer
    if dim < 1:
        raise ValueError("Argument 'dim' must be a positive integer.")

    if rank > 0:
        return sym.MutableDenseNDimArray(
            [0 for _ in range(dim**rank)],
            tuple(dim for _ in range(rank))
        )
    raise ValueError("Argument 'rank' must be non-negative.")


def elements(
    array: sym.Array
) -> Dict[Tuple[int, ...], sym.Array]:
    """
    Returns the elements of a sym.Array as a dictionary.
    """
    if not isinstance(array, sym.Array):
        raise TypeError("Argument 'array' must be a SymPy Array.")

    components = {}

    # Iterate through components array until a single element is returned
    # Appending it to the list of elements with the corresponding indices
    def recurse(arr: sym.Array, indices: Tuple[int, ...] = ()):
        if isinstance(arr, sym.Array) and arr.shape:
            for index, arr_iter in enumerate(arr):
                recurse(arr_iter, (*indices, index))
        else:
            components[indices] = arr
    recurse(arr=array)
    return components


def map_to_array(
    array: sym.Array,
    func: Callable[[Any], Any],
    *args: Any
) -> sym.Array:
    """
    Map a function to each element of an array.
    """

    if isinstance(array, sym.Matrix):
        array = sym.Array(array)

    if not isinstance(array, sym.Array):
        raise TypeError("Argument 'array' must be a sympy Array.")

    if not callable(func):
        raise TypeError("Argument 'func' must be callable.")

    # Create an array of zeros of an appropriate shape
    result = zero_tensor(
        dim=array.shape[0] if array.shape else 0,
        rank=array.rank()
    )

    # Retrieve the elements of the array
    array_elements = elements(array)

    # Loop over the elements, applying the function to each
    if array.rank() == 0:
        result = func(sym.Array(array_elements[()])[()], *args)

    if array.rank() != 0:
        for indices, element in array_elements.items():
            result[indices] = func(element, *args)

    return sym.Array(result)


def contract(
    array1: sym.Array,
    array2: sym.Array,
    *indices: Tuple[int, int]
) -> sym.Array:
    """
    Contract two arrays with respect to a set of indices.

    If indices are not supplied, the contraction is simply the tensor product of the two arrays.

    Each contraction index is assumed to have the first index be relative to the first array, and the second index be relative to the second array.
    """

    if isinstance(array1, sym.Matrix):
        array1 = sym.Array(array1)

    if isinstance(array2, sym.Matrix):
        array2 = sym.Array(array2)

    # Make sure user input is valid
    if not isinstance(array1, sym.Array):
        raise TypeError(
            f"Expected type '{sym.Array}' for argument 'array1', got '{type(array1)}'"
        )

    if not isinstance(array2, sym.Array):
        raise TypeError(
            f"Expected type '{sym.Array}' for argument 'array2', got '{type(array2)}'"
        )

    if not indices:
        return sym.Array(sym.tensorproduct(array1, array2))

    # Make sure indices are valid
    for index in indices:
        if index and any((i < 0 for i in index)):
            raise ValueError("Indices must be non-negative.")
        if index[0] >= array1.rank() or index[1] >= array2.rank():
            raise ValueError("Indices must be less than array rank.")
    _indices = [i for i, _ in indices]
    for i in range(array1.rank()):
        if _indices.count(i) > 1:
            raise ValueError("Indices must be unique.")
    _indices = [j for _, j in indices]
    for j in range(array2.rank()):
        if _indices.count(j) > 1:
            raise ValueError("Indices must be unique.")

    # Determine the shape of the resulting array
    shape = [*array1.shape, *array2.shape][0:-2 * len(indices)]

    # Create an array of zeros of an appropriate shape
    result = zero_tensor(
        shape[0] if shape else 0,
        len(shape)
    )

    # The possible indices of the resulting array
    all_indices = tuple(
        i for i in range(array1.shape[0])
    )

    # The indices over which we will be summing
    indices_to_sum = tuple(
        product(
            *tuple(
                all_indices for _ in range(len(indices))
            )
        )
    )

    # Create a list of slices for each index which will be summed over
    slices_array1 = []
    slices_array2 = []
    for index_pair in indices_to_sum:
        array1_index = [slice(0, None) for _ in range(len(array1.shape))]
        array2_index = [slice(0, None) for _ in range(len(array2.shape))]
        for i, index in enumerate(index_pair):
            array1_index[indices[i][0]] = index  # type: ignore[call-overload]
            array2_index[indices[i][1]] = index  # type: ignore[call-overload]
        slices_array1.append(array1_index)
        slices_array2.append(array2_index)

    # For each pair of array slices, sum the tensor products
    for slice_array1, slice_array2 in zip(slices_array1, slices_array2):
        contraction = sym.tensorproduct(
            array1[tuple(slice_array1)],
            array2[tuple(slice_array2)]
        )
        if not isinstance(contraction, sym.Array):
            contraction = sym.Array(contraction)
        result += contraction

    if shape:
        result = sym.Array(sym.simplify(result))
    if not shape:
        result = sym.Array(sym.simplify(result[()]))

    return result


def partial_contract(
    coordinates: sym.Array,
    array: sym.Array,
    index: Optional[int] = None
) -> sym.Array:
    """
    Take the partial derivative of an array with respect to the coordinates.

    If index is supplied, the divergence is calculated, otherwise, the gradient is calculated.
    """

    if not isinstance(coordinates, sym.Array):
        coordinates = sym.Array(coordinates)  # type: ignore[unreachable]

    if not isinstance(array, sym.Array):
        array = sym.Array(array)  # type: ignore[unreachable]

    # Make sure indices are valid
    if index is not None and index < 0:
        raise ValueError("Indices must be non-negative.")
    if index is not None and (index > coordinates.rank() or index > array.rank()):
        raise ValueError("Indices must be less than array rank.")

    # Determine the shape of the resulting array
    shape = [*coordinates.shape, *array.shape][0:-2 if index is not None else None]

    # Create an array of zeros of an appropriate shape
    result = zero_tensor(
        shape[0] if shape else 0,
        len(shape)
    )

    # If index is not supplied, calculate the gradient of the tensor
    if index is None:
        if array.rank() == 0:
            array = array[()]
        for i in range(coordinates.shape[0]):
            slice_array = [slice(0, None) for _ in range(len(result.shape))]
            slice_array[0] = i  # type: ignore[call-overload]
            result[tuple(slice_array)] += sym.diff(array, coordinates[i])

    # For each index, we loop over, then loop over the tensor taking the partial deriative and summing
    if index is not None:
        for i, coord in iter(enumerate(coordinates)):
            slice_array = [slice(0, None) for _ in range(len(array.shape))]
            slice_array[index] = i  # type: ignore[call-overload]
            temp = sym.diff(array[tuple(slice_array)], coord)
            result += temp if isinstance(temp, sym.Array) else sym.Array(temp)

    return sym.Array(result) if result.rank() else sym.Array(result[()])
