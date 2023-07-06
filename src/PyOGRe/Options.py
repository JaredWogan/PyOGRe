from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import sympy as sym
from IPython.display import Markdown, display

if TYPE_CHECKING:
    from PyOGRe.OGReObject import OGReObject


__doc__ = """
Options Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


@dataclass
class Options:
    """
    PyOGRe Options class.
    """

    LATEX: bool = True
    ALL_SYMBOLS: Tuple[sym.Symbol, ...] = sym.symbols(
        "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    )
    CURVE_PARAMETER: sym.Symbol = sym.Symbol("lambda")
    INFO_ORDER: Tuple[str, ...] = (
        "Name",
        "Symbol",
        "Type",
        "Rank",
        "Metric",
        "Default Coordinates",
        "Default Indices",
        "Default Coordinates For",
        "Coordinate Transformations",
        "Indices Symbols",
        "Tensors Using This Metric"
    )
    LIST_PER_LINE: int = 6
    FONT_SIZE: int = 14
    PARALLEL: bool = False


options = Options()


def set_parallelization(parallel: Optional[bool] = None) -> None:
    """
    This function will either print the existing parallelization mode, or will overwrite it if an argument is given.

    The Default value is False.

    Parallelization is currently WIP.
    """

    if parallel is None:
        print(options.PARALLEL)

    if parallel is not None and not isinstance(parallel, bool):
        raise TypeError(
            f"The argument 'parallel' must be a boolean, got {type(parallel)}."
        )

    if parallel is not None:
        options.PARALLEL = parallel
        if options.PARALLEL:
            pass
        if not options.PARALLEL:
            pass


def set_list_per_line(number: Optional[int] = None) -> None:
    """
    This function will either print the existing list per line, or will overwrite it if an argument is given.

    Controls how many symbols are printed per line in a Jupyter Notebook when calling the .list() method.

    The default value is 6.
    """

    if number is None:
        print(options.LIST_PER_LINE)

    if number is not None and not isinstance(number, int):
        raise TypeError(
            f"The argument 'number' must be an integer, got {type(number)}."
        )

    if number is not None and number < 0:
        raise ValueError(
            f"The argument 'number' must be a positive integer, got {number}."
        )

    if isinstance(number, int):
        options.LIST_PER_LINE = number


def set_font_size(size: Optional[int] = None) -> None:
    """
    This function will either print the existing font size, or will overwrite it if an argument is given.

    The default value is 14.
    """

    if size is None:
        print(options.FONT_SIZE)

    if size is not None and not isinstance(size, int):
        raise TypeError(
            f"The argument 'size' must be an integer, got {type(size)}."
        )

    if size is not None and size < 0:
        raise ValueError(
            f"The argument 'size' must be a positive integer, got {size}."
        )

    if isinstance(size, int):
        options.FONT_SIZE = size


def set_index_letters(letters: Optional[str] = None) -> None:
    """
    This function will either print the existing index letters, or will overwrite them if an argument is given.

    Supplying an argument of 'automatic' will set the index letters to the default.
    Supplying an argument of 'greek' will set the index letters to the greek alphabet.
    Supplying an argument of 'english' will set the index letters to the english alphabet.

    Expects the argument 'letters' to be of the form 'a b c d...'.

    >>> og.set_index_letters("i j k l m n o p")
    (i, j, k, l, m, n, o, p)

    >>> og.set_index_letters("automatic")
    (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z)
    """

    if letters is None:
        if options.LATEX:
            display(Markdown(
                "$" +
                sym.latex(options.ALL_SYMBOLS) +
                "$"
            ))
        if not options.LATEX:
            print(options.ALL_SYMBOLS)

    if letters is not None and not isinstance(letters, str):
        raise TypeError(
            f"The argument 'letters' must be a string, got {type(letters)}."
        )

    if letters is not None and letters.lower() == "automatic":
        options.ALL_SYMBOLS = sym.symbols(
            "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        )
        if options.LATEX:
            display(Markdown(
                "$" +
                sym.latex(options.ALL_SYMBOLS) +
                "$"
            ))
        if not options.LATEX:
            print(options.ALL_SYMBOLS)

    if letters is not None and letters.lower() == "greek":
        options.ALL_SYMBOLS = sym.symbols(
            "mu nu rho sigma kappa lambda alpha beta gamma delta epsilon zeta theta iota xi pi tau phi chi psi omega"
        )
        if options.LATEX:
            display(Markdown(
                "$" +
                sym.latex(options.ALL_SYMBOLS) +
                "$"
            ))
        if not options.LATEX:
            print(options.ALL_SYMBOLS)

    if letters is not None and letters.lower() == "english":
        options.ALL_SYMBOLS = sym.symbols(
            "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        )
        if options.LATEX:
            display(Markdown(
                "$" +
                sym.latex(options.ALL_SYMBOLS) +
                "$"
            ))
        if not options.LATEX:
            print(options.ALL_SYMBOLS)

    if letters is not None and letters.lower() not in ["automatic", "greek", "english"]:
        options.ALL_SYMBOLS = sym.symbols(letters)
        if options.LATEX:
            display(Markdown(
                "$" +
                sym.latex(options.ALL_SYMBOLS) +
                "$"
            ))
        if not options.LATEX:
            print(options.ALL_SYMBOLS)


def set_curve_parameter(symbol: Optional[Union[sym.Symbol, str]] = None) -> None:
    """
    (function) set_curve_parameter(symbol)

    This function will either print the existing curve parameter, or will overwrite it if an argument is given.

    `symbol`: The symbol to set as the curve parameter. Supplying an argument of 'automatic' will set the index letters to the default.
    """

    if symbol is None:
        if options.LATEX:
            display(Markdown(
                "$" +
                sym.latex(options.CURVE_PARAMETER) +
                "$"
            ))
        if not options.LATEX:
            print(options.CURVE_PARAMETER)

    if symbol is not None and not isinstance(symbol, (sym.Symbol, str)):
        raise TypeError(
            f"The argument 'symbol' must be a SymPy Symbol or 'Automatic', got {type(symbol)}."
        )

    if isinstance(symbol, str) and symbol.lower() == "automatic":
        options.CURVE_PARAMETER = sym.Symbol("lambda")
        if options.LATEX:
            display(Markdown(
                "$" +
                sym.latex(options.CURVE_PARAMETER) +
                "$"
            ))
        if not options.LATEX:
            print(options.CURVE_PARAMETER)

    if isinstance(symbol, str) and symbol.lower() != "automatic" or isinstance(symbol, sym.Symbol):
        if isinstance(symbol, str):
            symbol = sym.symbols(symbol, seq=True)[0]

        options.CURVE_PARAMETER = symbol

        if options.LATEX:
            display(Markdown(
                "$" +
                sym.latex(options.CURVE_PARAMETER) +
                "$"
            ))
        if not options.LATEX:
            print(options.CURVE_PARAMETER)


def get_curve_parameter() -> sym.Symbol:
    """
    This function will return the curve parameter.
    """

    return options.CURVE_PARAMETER


def jupyter_support() -> None:
    """
    This function is used to enable Jupyter Notebook support.
    """
    options.LATEX = True


def command_line_support() -> None:
    """
    This function is used to support Command Line.
    """
    options.LATEX = False


def get_options() -> None:
    """
    This function is used to get the current options set by the package or user.
    """
    for option in options.__dict__:
        print(f"{option}: {options.__dict__[option]}")


def get_instances() -> List[OGReObject]:
    """
    This function is used to get the current instances of the OGRE package.
    """
    from PyOGRe.OGReObject import OGReObject
    return OGReObject._instances


def delete_results(silent: bool = False) -> None:
    """
    This function is used to delete all tensors named 'Result', created from the Calc function.

    Note: this will not always remove every occurence of a tensor named 'Result'.
    """

    instances = get_instances()
    deleted = 0
    for _ in range(4):
        for instance in instances:
            if instance._name == "Result":
                try:
                    instance.delete(silent=silent)
                    deleted += 1
                except PermissionError:
                    pass

    if not silent:
        if options.LATEX:
            font_size = options.FONT_SIZE
            display(Markdown(
                f"<div align=left style='font-size:{font_size}pt; margin-bottom:12pt'> \n\n" +
                f"Deleted {deleted} tensor(s) named 'Result'."
                "\n\n </div>"
            ))

        if not options.LATEX:
            print(f"Deleted {deleted} tensor(s) named 'Result'.")


def clear_instances() -> None:
    """
    This function is used to clear all instances of the OGRE package.
    """
    from PyOGRe.OGReObject import OGReObject
    delete_results(silent=True)
    for _ in range(4):
        for instance in OGReObject._instances:
            try:
                instance.delete(silent=True)
            except PermissionError:
                pass
    OGReObject._instances = []
