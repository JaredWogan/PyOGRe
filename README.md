![Verified for Python 3.7](https://github.com/JaredWogan/PyOGRe/actions/workflows/python37.yml/badge.svg)
![Verified for Python 3.8](https://github.com/JaredWogan/PyOGRe/actions/workflows/python38.yml/badge.svg)
![Verified for Python 3.9](https://github.com/JaredWogan/PyOGRe/actions/workflows/python39.yml/badge.svg)
[![License: MIT](https://img.shields.io/github/license/JaredWogan/PyOGRe)](https://github.com/JaredWogan/PyOGRe/blob/master/LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/JaredWogan/PyOGRe)
[![GitHub repo stars](https://img.shields.io/github/stars/JaredWogan/PyOGRe?style=social)](https://github.com/JaredWogan/PyOGRe)
[![Twitter @JaredWogan](https://img.shields.io/twitter/follow/JaredWogan?style=social)](https://twitter.com/JaredWogan)
[![Twitter @BarakShoshany](https://img.shields.io/twitter/follow/BarakShoshany?style=social)](https://twitter.com/BarakShoshany)


[Open in Visual Studio Code](https://open.vscode.dev/JaredWogan/PyOGRe)

# PyOGRe - A Python Object-Oriented General Relativity Package

PyOGRe is the Python port of the Mathematica package OGRe, written by Professor Barak Shoshany at Brock University. OGRe is a tensor calculus package that is designed to be both powerful and user-friendly. OGRe can perform calculations involving tensors extremely quickly, which could often take hours to do by hand. It is extremely easy to pick up and use, with easy to learn syntax. Naturally, the package has applications in general relativity and differential geometry where tensors are used abundantly and was the focus point during the development of the package. However, there is no restriction preventing the package from being used for other applications.

## Installation
Currently the package is waiting for full release on PyPI. For now, git clone the repository and run `pip install .` from the root directory.

Subsequently, you may import the package just as you would any other package by using `import PyOGRe` or `import PyOGRe as og`

## Features
- Define coordinate systems and any transformation rules between them. Tensor components in any representation can then be transformed automatically as needed.
- Define metrics which can then be used to define any arbitrary tensor. The metric associated with a tensor will be used to raise and lower indices as needed.
- Display any tensor object in any coordinate system as an array, or as a list of the unique non-zero components. Metrics can additionally be displayed as a line element.
    - When displaying a tensor, substitutions can be made for any variable or function present. Additionally, a function may be specified that will be mapped to each component of the tensor.
- Export tensors to a file so that they can later be imported into a new session or into the Mathematica version.
- A simple API for performing calculations on tensors, including addition, subtraction, multiplication by a scalar, trace, contraction, as well as partial and covariant derivatives.
- Built-in tensors for commonly used coordinate systems and metrics.
- Built-in functions for calculating the Christoffel symbols (the Levi-Cevita connection), Riemann tensor, Ricci tensor, Ricci scalar, Einstein Tensor, curve Lagrangian, and volume element from a metric, as well as the norm squared of any tensor.
- Calculate the geodesic equations in terms of a user defined curve parameter (affine parameter), using either the Christoffel symbols or the curve Lagrangian (for spacetime metrics, the geodesic equations can be calculated in terms of the time coordinate).
- Designed to be performant using optimized algorithms for common operations (these functions can be used on any SymPy array).
- Quick and easy to install straight from PyPI (supports Python 3.6 and above, previous versions untested).
- Command line and Jupyter notebook support.
- Clear and well documented source code, complete with examples.
- Open source and available for all to use.
- Easily extendible and modifiable.
- Under active development and will be updated regularly.
