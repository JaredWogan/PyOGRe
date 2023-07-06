import copy

import PyOGRe as og
import pytest
import sympy as sym


def test_new_tensor():
    """
    Tests PyOGRe.Tensor.new_tensor()
    """

    t, x, y, z, r, theta, phi = sym.symbols("t x y z r theta phi", seq=True)
    cartesian = og.new_coordinates(
        **dict(
            name="Cartesian",
            components=sym.Array([t, x, y, z]),
            indices=(1,),
            symbol="x",
            transformations=None
        )
    )
    spherical = og.new_coordinates(
        **dict(
            name="Spherical",
            components=sym.Array([t, r, theta, phi]),
            indices=(1,),
            symbol="x",
            transformations=None
        )
    )
    cartesian.add_coord_transformation(
        coords=spherical,
        rules=[
            None,
            sym.Eq(x, r*sym.sin(theta)*sym.cos(phi)),
            sym.Eq(y, r*sym.sin(theta)*sym.sin(phi)),
            sym.Eq(z, r*sym.cos(theta))
        ]
    )
    spherical.add_coord_transformation(
        coords=cartesian,
        rules=[
            sym.Eq(r, sym.sqrt(x**2+y**2+z**2)),
            sym.Eq(theta, sym.acos(z/(sym.sqrt(x**2+y**2+z**2)))),
            sym.Eq(phi, sym.atan(y/x)),
        ]
    )
    minkowski_kwargs = dict(
        name="Minkowski",
        components=sym.diag(-1, 1, 1, 1),
        indices=(-1, -1),
        symbol="g"
    )
    minkowski = cartesian.new_metric(**minkowski_kwargs)

    scalar_kwargs = dict(
        name="Scalar",
        components=sym.Array(42*x*y**2*z**3*sym.sqrt(t)),
        indices=(),
        symbol="S"
    )
    scalar = minkowski.new_tensor(**scalar_kwargs)

    tensor_kwargs = dict(
        name="Tensor",
        components=sym.diag(t*x*y*z, sym.sin(x), x-y, z**t),
        indices=(1, 1),
        metric=minkowski,
        coords=cartesian,
        symbol="T"
    )
    tensor = og.new_tensor(**tensor_kwargs)

    with pytest.raises(TypeError):
        bad_kwargs = copy.deepcopy(tensor_kwargs)
        bad_kwargs["metric"] = "Not a metric"
        minkowski.new_tensor(**bad_kwargs)

    with pytest.raises(TypeError):
        tensor(42)

    assert scalar.rank() == 0
    assert tensor.rank() == 2
    assert tensor.info() is None
    assert cartesian.info() is None  # Additional coverage for info() method on Coordinate Objects
