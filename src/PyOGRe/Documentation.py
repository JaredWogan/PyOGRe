from PyOGRe.Version import __version__

__doc__ = f"""
PyOGRe Documentation

Version: {__version__}

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


def doc() -> None:
    """
    Prints the documentation to the console.
    """
    print(__doc__)
