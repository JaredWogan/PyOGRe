import pytest
import sympy as sym
import PyOGRe as og


def test_new_coordinates():
    """
    Tests PyOGRe.Coordinates.new_coordinates()
    """

    cartesian_symbols = sym.symbols("t, x, y, z")

    proper_kwargs = dict(
        name="Cartesian",
        components=sym.Array(cartesian_symbols),
        indices=(1,),
        symbol="x",
        transformations=None
    )
    cartesian = og.new_coordinates(**proper_kwargs)

    with pytest.raises(TypeError):
        test_kwargs = proper_kwargs.copy()
        test_kwargs["name"] = 42
        og.new_coordinates(**test_kwargs)

    with pytest.raises(TypeError):
        test_kwargs = proper_kwargs.copy()
        test_kwargs["components"] = "Not an array"
        og.new_coordinates(**test_kwargs)

    with pytest.raises(TypeError):
        test_kwargs = proper_kwargs.copy()
        test_kwargs["symbol"] = 42
        og.new_coordinates(**test_kwargs)

    with pytest.raises(Exception):
        test_kwargs = proper_kwargs.copy()
        test_kwargs["components"] = sym.Array(
            [
                [1, 2], [3, 4], [5, 6], [7, 8]
            ]
        )
        og.new_coordinates(**test_kwargs)

    with pytest.raises(Exception):
        test_kwargs = proper_kwargs.copy()
        test_kwargs["components"] = sym.Array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        og.new_coordinates(**test_kwargs)

    with pytest.raises(TypeError):
        test_kwargs = proper_kwargs.copy()
        test_kwargs["indices"] = "Not a tuple"
        og.new_coordinates(**test_kwargs)

    with pytest.raises(Exception):
        test_kwargs = proper_kwargs.copy()
        test_kwargs["indices"] = (1, 1)
        og.new_coordinates(**test_kwargs)

    with pytest.raises(NotImplementedError):
        og.new_coordinates(**proper_kwargs)(42)

    assert cartesian.get_components() == sym.Array(cartesian_symbols)
    assert cartesian.get_components(mode="mathematica") == "{t, x, y, z}"
    assert cartesian.get_components(mode="latex") == r"\left(\begin{matrix}t\\x\\y\\z\end{matrix}\right)"
    assert cartesian.show() is None
    assert cartesian.list() is None
    assert cartesian.info() is None

    from PyOGRe.Defaults import spherical
    with pytest.raises(Exception):
        cartesian.add_coord_transformation(
            coords="Not a Coordinates object",
            rules=[None, None, None, None]
        )
    with pytest.raises(Exception):
        cartesian.add_coord_transformation(
            coords=cartesian,
            rules=[None, None, None, None]
        )
    with pytest.raises(Exception):
        cartesian.add_coord_transformation(
            coords=spherical,
            rules=[None, 42, None, None]
        )

    assert cartesian.add_coord_transformation(
        spherical,
        [None, None, None, None]
    ) is cartesian
    assert cartesian.add_coord_transformation(
        spherical,
        None
    ) is cartesian
