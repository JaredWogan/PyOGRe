from __future__ import annotations

from abc import ABC, abstractmethod
from gc import collect
from typing import Dict, List, Optional, Tuple, Union

import sympy as sym
from IPython.display import Markdown, display

from PyOGRe.Exceptions import (IndicesDimensionError, IndicesValueError,
                               TensorDimensionError)
from PyOGRe.Options import options


__doc__ = """
OGReObject Module: Abstract Base Class

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


class OGReObject(ABC):
    """
    Abstract Base Class of all PyOGRe Objects. An instance of this class cannot be created, but
    all objects in PyOGRe inherit from this class. Objects in PyOGRe should not be accessed
    directly, but instead the defined methods should be called on them.
    """

    _instances: List[OGReObject] = []

    def __init__(
        self: OGReObject,
        name: str,
        components: sym.Array,
        indices: Optional[Tuple[int, ...]] = None,
        symbol: Union[str, sym.Symbol] = sym.Symbol("T")
    ) -> None:
        """
        OGReObject Constructor. Validates all user input to ensure that the Tensor is valid.
        """

        # Run the OGReObject Validation process
        ValidateOGReObject(
            name=name,
            components=components,
            symbol=symbol,
            indices=indices
        )

        # Initialize the Tensor
        self.__class__._instances.append(self)
        self._name = name
        self._components: Dict[object, Dict[Tuple[int, ...], sym.Array]] = {}
        self._rank = components.rank()

        if isinstance(symbol, str):
            self._symbol = sym.Symbol(symbol) if symbol != "" else symbol

        if isinstance(symbol, sym.Symbol):
            self._symbol = symbol

        self._indices = indices if indices is not None else tuple(-1 for _ in range(self._rank))

    # ===================
    # Dunder Methods
    # ===================

    def __str__(
        self: OGReObject
    ) -> str:
        """
        str() Method
        Used to print the Tensor
        """

        info = self._get_info()

        string_representation = ""
        for key in options.INFO_ORDER:
            if key in info:
                string_representation += f"{key}: {info[key]} \n"

        return string_representation

    def __repr__(
        self: OGReObject
    ) -> str:
        """
        repr() Method
        Used to print the Tensor
        """
        return self._name

    def __hash__(
        self: OGReObject
    ) -> int:
        """
        hash() Method
        Used to hash the Tensor
        Combined with the __eq__ method, this allows for the Tensor to be used as a key in a dictionary
        """
        return id(self)

    def __eq__(
        self: OGReObject,
        other: object
    ) -> bool:
        """
        == Method
        Combined with the __hash__ method, this allows for the Tensor to be used as a key in a dictionary
        """

        if not isinstance(other, OGReObject):
            return False

        return id(self) == id(other)

    # ===================
    # Private Methods
    # ===================

    def _get_info(
        self: OGReObject,
    ) -> Dict[str, str]:
        """
        Gets the information of the OGReObject.
        """
        info = {
            "Name": self._name,
            "Symbol": "$" + sym.latex(self._symbol) + "$" if options.LATEX else str(self._symbol),
            "Type": self.__class__.__name__,
            "Rank": self._rank,
            "Default Indices": self._indices
        }
        return info

    def _eq_symbol(
        self: OGReObject,
        indices: Tuple[int, ...]
    ) -> str:
        """
        """

        indices_symbols = options.ALL_SYMBOLS[:self._rank]

        if options.LATEX:
            symbol = sym.latex(self._symbol) if self._symbol != "" else r"\square"
            for i, index in enumerate(indices):
                if index == -1:
                    symbol += "_{" + sym.latex(indices_symbols[i]) + "}{ }"
                if index == 1:
                    symbol += "^{" + sym.latex(indices_symbols[i]) + "}{ }"
            return str(symbol)

        if not options.LATEX:
            symbol = str(self._symbol) if self._symbol != "" else "T"
            for i, index in enumerate(indices):
                if index == -1:
                    symbol += "_" + str(indices_symbols[i])
                if index == 1:
                    symbol += "^" + str(indices_symbols[i])
            return str(symbol)

        return ""

    # ===================
    # Public Methods
    # ===================

    def export(
        self: OGReObject,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Exports a tensor to a string.

        Supplying a filepath as a string will write the tensor data to a file.
        """
        if filename is not None and not isinstance(filename, str):
            raise TypeError(
                f"Expected type '{str}' for argument 'filename', got '{type(filename)}'."
            )

        if filename is None:
            from PyOGRe.Export import export_to_string
            return export_to_string(self)

        if filename is not None:
            from PyOGRe.Export import export_to_file
            status = export_to_file(self, filename)
            if options.LATEX:
                display(Markdown(
                    "<div align=left style='font-size:12pt'> \n\n" +
                    status +
                    "\n\n </div>"
                ))
            if not options.LATEX:
                print(status)

        return None

    def delete(
        self: OGReObject,
        silent: bool = False
    ) -> None:
        """
        Deletes the Tensor.
        """

        for instance in OGReObject._instances:
            if instance is not self:
                if self in instance._components:
                    raise PermissionError(
                        f"Cannot delete {self._name} because it is still used in another Tensor ({instance!r})."
                    )

                if hasattr(instance, "_metric"):
                    if getattr(instance, "_metric") == self:
                        raise PermissionError(
                            f"Cannot delete {self._name} because it is still used as a Metric for another Tensor ({instance!r})."
                        )

                if hasattr(instance, "_coordinates"):
                    if getattr(instance, "_coordinates") == self:
                        raise PermissionError(
                            f"Cannot delete {self._name} because it is still used as the Coordinates for another Tensor ({instance!r})."
                        )

                if hasattr(instance, "_transformations"):
                    if self in getattr(instance, "_transformations"):
                        raise PermissionError(
                            f"Cannot delete {self._name} because it is still used as a Transformation for another Tensor ({instance!r})."
                        )

        if not silent:
            if options.LATEX:
                font_size = options.FONT_SIZE
                display(Markdown(
                    f"<div align=left style='font-size:{font_size + 2}pt; margin-bottom:12pt'> \n\n" +
                    f"__{self._name}:__" +
                    "\n\n </div>" +
                    f"<div align=left style='font-size:{font_size}pt'> \n\n" +
                    "Successfully Deleted"
                    "\n\n </div>"
                ))

            if not options.LATEX:
                print(f"{self._name}: Successfully Deleted.")

        for _ in range(3):
            for attr in dir(self):
                if attr.startswith("_") and not attr.startswith("__"):
                    try:
                        delattr(self, attr)
                    except AttributeError:
                        pass

        self.__class__._instances.remove(self)
        del self
        collect()

    def info(
        self: OGReObject
    ) -> None:
        """
        Displays the Tensor information
        """

        info = self._get_info()

        if options.LATEX:
            size = options.FONT_SIZE
            display(Markdown(
                f"<div align=left style='font-size:{size+2}pt; margin-bottom:12pt'> \n\n" +
                f"{self._name}:" +
                "\n\n </div>" +
                f"<div align=left style='font-size:{size-2}pt'> \n\n" +
                "\n\n".join(
                    [
                        f"__{key}:__ {info[key]}"
                        for key in options.INFO_ORDER if key in info
                    ]
                ) +
                "\n\n </div>"
            ))

        if not options.LATEX:
            sym.pprint(f"{self._name}:")
            for key in options.INFO_ORDER:
                if key in info:
                    sym.pprint(f"{key}: {info[key]}")

    def rank(
        self: OGReObject
    ) -> int:
        """
        Returns the rank of the OGReObject.
        """
        return int(self._rank)

    def change_name(
        self: OGReObject,
        name: str
    ) -> OGReObject:
        """
        Changes the name of the Tensor.
        """

        if not isinstance(name, str):
            raise TypeError(
                f"Expected type '{str}' for argument 'components', got '{type(name)}'."
            )

        self._name = name

        return self

    def change_symbol(
        self: OGReObject,
        symbol: Union[sym.Symbol, str]
    ) -> OGReObject:
        """
        Changes the symbol used to represent the Tensor.
        """

        if isinstance(symbol, str):
            symbol = sym.Symbol(symbol)

        if not isinstance(symbol, sym.Symbol):
            raise TypeError(
                f"Expected type '{sym.Symbol}' for argument 'components', got '{type(symbol)}'."
            )

        self._symbol = symbol

        return self

    # ===================
    # Abstract Methods
    # ===================

    @abstractmethod
    def show(
        self: OGReObject
    ) -> None:
        """
        Displays the Tensor symbolically.
        """

    @abstractmethod
    def list(
        self: OGReObject
    ) -> None:
        """
        Displays the non-zero elements of the Tensor component wise.
        """

    @abstractmethod
    def get_components(
        self: OGReObject,
        mode: str = "sympy"
    ) -> Union[sym.Array, str]:
        """
        Returns the components of the Tensor in the requested representation.
        """


def ValidateIndices(
    indices: Optional[Tuple[int, ...]] = None,
    components: Optional[sym.Array] = None,
    source_indices: Optional[Tuple[int, ...]] = None
) -> bool:
    """
    Validates supplied indices.
    """

    # Check to see if indices is a tuple
    if indices is not None and not isinstance(indices, tuple):
        raise TypeError(
            f"Expected type '{tuple}' for argument 'indices', got '{type(indices)}'."
        )

    # Check to make sure all indices are either +1 or -1
    if indices is not None:
        for index in indices:
            if index not in (-1, 1):
                raise IndicesValueError(
                    indices=indices
                )

    # If the number of indices doesn't match the number of dimensions of the Tensor
    if components is not None and indices is not None:
        if isinstance(components, sym.Array) and components.rank() != len(indices):
            raise IndicesDimensionError(
                indices=indices,
                error="tensor",
                dim=components.rank()
            )
        if isinstance(components, sym.Number) and len(indices) != 0:
            raise IndicesDimensionError(
                indices=indices,
                error="tensor",
                dim=0
            )

    # If the number of indices doesn't match the coords number of indices
    if source_indices is not None and indices is not None and len(source_indices) != len(indices):
        raise IndicesDimensionError(
            indices=indices,
            error="symbol",
            dim=len(source_indices)
        )

    return True


def ValidateOGReObject(
    name: str,
    components: sym.Array,
    symbol: Optional[Union[str, sym.Symbol]],
    indices: Optional[Tuple[int, ...]]
) -> bool:
    """
    Validates an OGReObject.
    """

    # Make sure name is a string
    if not isinstance(name, str):
        raise TypeError(
            f"Expected type '{str}' for argument 'name', got '{type(name)}'."
        )

    # Make sure the symbol is a single SymPy.Symbol
    if not isinstance(symbol, sym.Symbol) and not isinstance(symbol, str):
        raise TypeError(
            f"Expected type '{sym.Symbol}' or type '{str}' for argument 'symbol', got '{type(symbol)}'."
        )

    # Make sure the components are a SymPy.Array
    if not isinstance(components, sym.Array):
        raise TypeError(
            f"Expected type '{sym.Array}' for argument 'components', got '{type(components)}'."
        )

    for i in range(components.rank()):
        if components.shape[i] != components.shape[0]:
            raise TensorDimensionError(
                tensor_shape=components.shape
            )

    # Make sure the indices are a tuple
    if indices is not None and not isinstance(indices, tuple):
        raise TypeError(
            f"Expected type '{tuple}' for argument 'indices', got '{type(indices)}'."
        )

    ValidateIndices(indices=indices, components=components)

    return True
