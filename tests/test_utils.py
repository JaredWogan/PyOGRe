import random as rng
import pytest
import sympy as sym
import PyOGRe as og


def test_str_symbols():
    """
    Tests PyOGRe.Utils.str_symbols()
    """
    with pytest.raises(TypeError):
        og.str_symbols(42)

    assert og.str_symbols("a b c") == "a b c"

    assert og.str_symbols("a b c d") == "a b c d"

    assert og.str_symbols("x0:10") == "x0 x1 x2 x3 x4 x5 x6 x7 x8 x9"


def test_zero_tensor():
    """
    Tests PyOGRe.Utils.zero_tensor()
    """
    with pytest.raises(TypeError):
        og.zero_tensor("Not a number", 3)

    with pytest.raises(TypeError):
        og.zero_tensor(0, "Not a number")

    with pytest.raises(ValueError):
        og.zero_tensor(0, 1)

    with pytest.raises(ValueError):
        og.zero_tensor(1, -1)

    assert og.zero_tensor(1, 0) == sym.Array(0)

    assert og.zero_tensor(2, 1) == sym.Array([0, 0])

    assert og.zero_tensor(3, 2) == sym.Array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_elements():
    """
    Tests PyOGRe.Utils.elements()
    """
    with pytest.raises(TypeError):
        og.elements(42)

    array, result = [], {}
    for i in range(rng.randint(1, 10)):
        number = rng.randint(1, 10)
        array.append(number)
        result[(i,)] = number
    assert og.elements(sym.Array(array)) == result
    arrayt = sym.Array(
        [
            [1, 2, 3],
            [9, 8, 7],
            [42, 117, 314]
        ]
    )

    assert og.elements(arrayt) == {
        (0, 0): 1,
        (0, 1): 2,
        (0, 2): 3,
        (1, 0): 9,
        (1, 1): 8,
        (1, 2): 7,
        (2, 0): 42,
        (2, 1): 117,
        (2, 2): 314
    }


def test_map_to_array():
    """
    Tests PyOGRe.Utils.map_to_array()
    """
    def f1(x):
        return x+1

    def f2(x):
        return x**2

    with pytest.raises(TypeError):
        og.map_to_array(42, f1)

    with pytest.raises(TypeError):
        og.map_to_array(sym.Array(2), 42)

    assert og.map_to_array(
        sym.Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]),
        f1
    ) == sym.Array([
        [2, 3, 4],
        [5, 6, 7],
        [8, 9, 10]
    ])

    assert og.map_to_array(
        sym.Array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]),
        f2
    ) == sym.Array([
        [1, 4, 9],
        [16, 25, 36],
        [49, 64, 81]
    ])

    assert og.map_to_array(sym.Array(2), f1) == sym.Array(3)


def test_contract():
    """
    Tests PyOGRe.Utils.contract()
    """
    with pytest.raises(TypeError):
        og.contract(sym.Array(1), 42)

    with pytest.raises(TypeError):
        og.contract(42, sym.Array(2))

    a = sym.Array([[1, 2, 3], [-4, 5, -6], [7, 8, 9]])
    b = sym.Array([[-1, 1, -1], [1, -1, 1], [-1, 1, -1]])
    c = sym.Array(
        [
            sym.Array(sym.diag(3, 4, 0, 1)),
            sym.Array(sym.zeros(4, 4)),
            sym.Array(sym.zeros(4, 4)),
            sym.Array(sym.zeros(4, 4))
        ]
    )
    d = sym.Matrix([[1, 2], [3, 4]])
    e = sym.Matrix([[0, 1], [-1, 2]])

    with pytest.raises(ValueError):
        og.contract(a, b, (-1, 1))

    with pytest.raises(ValueError):
        og.contract(a, b, (2, 1))

    with pytest.raises(ValueError):
        og.contract(a, b, (1, 1), (1, 0))

    with pytest.raises(ValueError):
        og.contract(a, b, (1, 1), (0, 1))

    assert og.contract(a, b, (0, 0), (1, 1)) == sym.Array(-25)

    assert og.contract(c, c, (0, 1), (1, 2), (2, 0)) == sym.Array(9)

    assert og.contract(a, b, (0, 1)) == sym.Array(
        [
            [-12, 12, -12],
            [-5, 5, -5],
            [-18, 18, -18]
        ]
    )

    assert og.contract(d, e) == sym.Array(
        [
            [[[0, 1], [-1, 2]], [[0, 2], [-2, 4]]],
            [[[0, 3], [-3, 6]], [[0, 4], [-4, 8]]]
        ]
    )


def test_partial_contract():
    """
    Tests PyOGRe.Utils.partial_contract()
    """
    t, x, y, z = sym.symbols("t x y z")
    coords_components = [t, x, y, z]
    coords = sym.Array(coords_components)
    array1_components = [
        [t**2, 2*t*x, 3*t*y, 4*t*z],
        [x*t, x**2, x*y, x*z],
        [y*t, y*x, y**2, y*z],
        [z*t, z*x, z*y, z**2]
    ]
    array1 = sym.Array(array1_components)
    array2 = sym.Array(t*x**2*y**3*z**4)

    with pytest.raises(ValueError):
        og.partial_contract(coords, array1, -1)

    with pytest.raises(ValueError):
        og.partial_contract(coords, array1, 4)

    assert og.partial_contract(coords, array1, None) == og.partial_contract(coords_components, array1_components, None)

    assert og.partial_contract(coords_components, array2, None) == sym.Array(
        [
            x**2*y**3*z**4,
            2*t*x*y**3*z**4,
            3*t*x**2*y**2*z**4,
            4*t*x**2*y**3*z**3
        ]
    )

    assert og.partial_contract(coords, array1_components, 0) == sym.Array([5*t, 6*x, 7*y, 8*z])

    assert og.partial_contract(coords, array1, 1) == sym.Array(
        [
            11*t,
            5*x,
            5*y,
            5*z
        ]
    )
