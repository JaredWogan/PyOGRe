import copy
import random as rng

import PyOGRe as og
import pytest
import sympy as sym


def test_new_metric():
    """
    Tests PyOGRe.Metric.new_metric()
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

    random_kwargs = dict(
        name="Minkowski",
        components=sym.Array(
            sym.diag(*[rng.randint(1, 10) for _ in range(4)])
        ),
        indices=(-1, -1),
        symbol="random"
    )
    random_metric = cartesian.new_metric(**random_kwargs)

    with pytest.raises(TypeError):
        bad_kwargs = copy.deepcopy(minkowski_kwargs)
        bad_kwargs["components"] = "Not an array"
        cartesian.new_metric(**bad_kwargs)

    with pytest.raises(ValueError):
        bad_kwargs = copy.deepcopy(minkowski_kwargs)
        bad_kwargs["indices"] = (-1, 1)
        cartesian.new_metric(**bad_kwargs)

    with pytest.raises(Exception):
        bad_kwargs = copy.deepcopy(minkowski_kwargs)
        bad_kwargs["indices"] = (-1, 1, 1)
        cartesian.new_metric(**bad_kwargs)

    with pytest.raises(Exception):
        test_kwargs = minkowski_kwargs.copy()
        test_kwargs["components"] = sym.Array(
            [sym.symbols("a, b, c, d")]
        )
        cartesian.new_metric(**test_kwargs)

    with pytest.raises(Exception):
        test_kwargs = minkowski_kwargs.copy()
        test_kwargs["components"] = sym.Array(
            [
                [-1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
        cartesian.new_metric(**test_kwargs)

    with pytest.raises(Exception):
        test_kwargs = minkowski_kwargs.copy()
        test_kwargs["indices"] = (1, 1, 1)
        cartesian.new_metric(**test_kwargs)

    with pytest.raises(TypeError):
        minkowski(42)

    with pytest.raises(TypeError):
        minkowski.line_element(coords=42)

    with pytest.raises(TypeError):
        minkowski.volume_element(coords=42)

    assert minkowski.get_components() == og.new_metric(coords=cartesian, **minkowski_kwargs).get_components()

    assert minkowski.get_components() == sym.Array(sym.diag(-1, 1, 1, 1))

    assert minkowski.get_components(indices=(1, 1)) == sym.Array(sym.diag(-1, 1, 1, 1))

    assert minkowski.get_components(indices=(1, 1)) == sym.Array(sym.diag(-1, 1, 1, 1))

    assert minkowski.get_components(indices=(1, -1)) == sym.Array(sym.diag(1, 1, 1, 1))

    assert random_metric.get_components(indices=(-1, 1)) == sym.Array(sym.diag(1, 1, 1, 1))

    assert minkowski.get_components(coords=spherical) == sym.Array(sym.diag(-1, 1, r**2, r**2*sym.sin(theta)**2))

    assert minkowski.get_components(coords=spherical, indices=(1, 1)) == sym.Array(sym.diag(-1, 1, 1/(r**2), 1/(r**2*sym.sin(theta)**2)))

    assert minkowski.show() is None

    assert minkowski.list() is None

    assert minkowski.info() is None
