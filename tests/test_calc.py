import pytest
import sympy as sym
import PyOGRe as og
from PyOGRe.Defaults import cartesian, spherical, minkowski, schwarzschild, flrw

a1, a2, a3, a4, s = sym.symbols("a b c d s", seq=True)
tensor = minkowski.new_tensor(
    name="Test Tensor",
    components=sym.diag(a1, a2, a3, a4),
    indices=(1, 1)
)
scalar = minkowski.new_tensor(
    name="Test Scalar",
    components=sym.Array(s),
    indices=()
)


def test_norm_squared():
    """
    Tests PyOGRe.BaseTensor.norm_squared()
    """

    assert minkowski.calc_norm_squared().get_components() == sym.Array(4)
    assert minkowski.calc_norm_squared().get_components() == minkowski.calc_norm_squared().get_components(coords=cartesian)
    assert schwarzschild.calc_norm_squared().get_components() == sym.Array(4)
    assert flrw.calc_norm_squared().get_components() == sym.Array(4)
    assert tensor.calc_norm_squared().get_components() == sym.Array(a1**2+a2**2+a3**2+a4**2)
    assert scalar.calc_norm_squared().get_components() == sym.Array(s**2)


def test_metric_line_element():
    """
    Tests PyOGRe.Metric.line_element()
    """

    assert minkowski.line_element() is None
    assert minkowski.line_element(coords=spherical) is None
    assert schwarzschild.line_element() is None
    assert flrw.line_element() is None


def test_metric_volume_element():
    """
    Tests PyOGRe.Metric.line_element()
    """

    assert minkowski.volume_element() is None
    assert minkowski.volume_element(coords=spherical) is None
    assert schwarzschild.volume_element() is None
    assert flrw.volume_element() is None


def test_add_subtract():
    """
    Tests PyOGRe.Calc.CalcObject add (+) and subtract (-) methods
    """

    anti_components = sym.Array(
        [
            [1, 0, 0, 1],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 4]
        ]
    )
    anti = minkowski.new_tensor(
        name="Anti Symmetric Test Tensor",
        components=anti_components,
        indices=(1, 1)
    )

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") + schwarzschild("a b"))

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") - schwarzschild("a b"))

    with pytest.raises(ValueError):
        og.Calc(og.PartialD("a") + og.PartialD("a"))

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") + minkowski("c d"))

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") - minkowski("c d"))

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") + scalar())

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") - scalar())

    with pytest.raises(ValueError):
        og.Calc(-og.PartialD("a"))

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") - og.PartialD("a"))

    assert og.Calc(scalar() + scalar()).get_components() == sym.Array(2*s)
    assert og.Calc(scalar() - scalar()).get_components() == sym.Array(0)
    assert og.Calc(tensor("a b") + tensor("a b")).get_components() == sym.Array(sym.diag(2*a1, 2*a2, 2*a3, 2*a4))
    assert og.Calc(0 + tensor("a b") - 0).get_components() == tensor.get_components()
    assert og.Calc(tensor("a b") + minkowski("a b")).get_components() == sym.Array(sym.diag(a1-1, a2+1, a3+1, a4+1))
    assert og.Calc((minkowski("a b") + anti("a b")) - (minkowski("a b") + anti("b a")), indices="a b").get_components(indices=(1, 1)) == sym.Array(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 0]
        ]
    )


def test_scalar_multiply():
    """
    Tests PyOGRe.Calc.CalcObject multiplication (*) methods
    """

    with pytest.raises(ValueError):
        og.Calc(og.PartialD("a") * minkowski("a b"))

    with pytest.raises(TypeError):
        og.Calc([1, 2] * minkowski("a b"))

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") * minkowski("a b"))

    assert og.Calc(s * scalar()).get_components() == sym.Array(s**2)
    assert og.Calc(s * scalar()).get_components() == og.Calc(scalar() * s).get_components()
    assert og.Calc(s * tensor("a b")).get_components() == sym.Array(sym.diag(s*a1, s*a2, s*a3, s*a4))
    assert og.Calc((s**2-1) * minkowski("a b")).get_components() == sym.Array(sym.diag(-(s**2-1), (s**2-1), (s**2-1), (s**2-1)))
    assert og.Calc(scalar() * tensor("a b")).get_components() == sym.Array(sym.diag(s*a1, s*a2, s*a3, s*a4))


def test_true_div():
    """
    Tests PyOGRe.Calc.CalcObject division (/) methods
    """

    with pytest.raises(ValueError):
        og.Calc(og.PartialD("a") / minkowski("a b"))

    with pytest.raises(TypeError):
        og.Calc([1, 2] / minkowski("a b"))

    with pytest.raises(TypeError):
        og.Calc(minkowski("a b") / minkowski("a b"))

    assert og.Calc(scalar() / s).get_components() == sym.Array(1)


def test_contraction():
    """
    Tests PyOGRe.Calc.CalcObject matmul (@) methods (tensor contraction)
    """

    anti_components = sym.Array(
        [
            [1, 0, 0, 1],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 4]
        ]
    )
    anti = minkowski.new_tensor(
        name="Anti Symmetric Test Tensor",
        components=anti_components,
        indices=(1, 1)
    )

    with pytest.raises(TypeError):
        og.Calc([1, 2] @ minkowski("a b"))

    with pytest.raises(ValueError):
        og.Calc(og.PartialD("a") @ og.CovariantD("a"))

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b") @ schwarzschild("a b"))

    assert og.Calc(scalar() @ scalar()).get_components() == sym.Array(s**2)
    assert og.Calc(tensor("a b") @ tensor("a b")).get_components() == sym.Array(a1**2+a2**2+a3**2+a4**2)
    assert og.Calc(tensor("a b") @ tensor("a c")).get_components() == sym.Array(
        [
            [-a1**2, 0, 0, 0],
            [0, a2**2, 0, 0],
            [0, 0, a3**2, 0],
            [0, 0, 0, a4**2]
        ]
    )
    assert og.Calc(anti("a b") @ tensor("a b")).get_components() == sym.Array(a1+2*a2+3*a3+4*a4)
    assert og.Calc(minkowski("a a")).get_components() == sym.Array(4)


def test_partial():
    """
    Tests PyOGRe.Calc.PartialD used in calculations
    """

    from PyOGRe.Calc import PartialD

    t, x, y, z = sym.symbols("t x y z", seq=True)
    var = minkowski.new_tensor(
        name="Variable Tensor",
        components=sym.Array(
            [t*x*y*z, t*x**2*y**3*z**4, sym.sin(t*x*y*z), sym.exp(t+x+y+z)]
        ),
        indices=(1,)
    )

    with pytest.raises(ValueError):
        og.Calc(PartialD("a b") @ scalar())

    assert og.Calc(PartialD("a") @ scalar()).get_components() == sym.Array([0, 0, 0, 0])
    assert sym.simplify(
        og.Calc(PartialD("a") @ var("a")).get_components()[()] -
        (sym.exp(t+x+y+z)+x*z*(y+2*t*y**3*z**3+t*sym.cos(t*x*y*z)))
    ) == 0
    assert og.Calc(PartialD("a") @ var("b")).get_components() == sym.Array(
        [
            [x*y*z, x**2*y**3*z**4, x*y*z*sym.cos(t*x*y*z), sym.exp(t+x+y+z)],
            [t*y*z, 2*t*x*y**3*z**4, t*y*z*sym.cos(t*x*y*z), sym.exp(t+x+y+z)],
            [t*x*z, 3*t*x**2*y**2*z**4, t*x*z*sym.cos(t*x*y*z), sym.exp(t+x+y+z)],
            [t*x*y, 4*t*x**2*y**3*z**3, t*x*y*sym.cos(t*x*y*z), sym.exp(t+x+y+z)]
        ]
    )
    assert og.Calc(PartialD("a") @ var("b")).get_components() == og.Calc(var("b") @ PartialD("a")).get_components()
    assert og.Calc(PartialD("a") @ var("a")).get_components() == og.Calc(var("a") @ PartialD("a")).get_components()


def test_covariant():
    """
    Tests PyOGRe.Calc.CovariantD used in calculations
    """

    from PyOGRe.Calc import CovariantD

    t, x, y, z, r, theta = sym.symbols("t x y z r theta", seq=True)
    var1 = minkowski.new_tensor(
        name="Variable Tensor 1",
        components=sym.Array(
            [t*x*y*z, t*x**2*y**3*z**4, sym.sin(t*x*y*z), sym.exp(t+x+y+z)]
        ),
        indices=(1,)
    )
    var2 = schwarzschild.new_tensor(
        name="Variable Tensor 2",
        components=sym.Array(
            [r, t*r, t*sym.cos(r), sym.exp(theta)]
        ),
        indices=(1,)
    )

    with pytest.raises(ValueError):
        og.Calc(CovariantD("a b") @ scalar())

    assert og.Calc(CovariantD("a") @ scalar()).get_components() == sym.Array([0, 0, 0, 0])
    assert sym.simplify(
        og.Calc(CovariantD("a") @ var1("a")).get_components()[()] -
        (sym.exp(t+x+y+z)+x*z*(y+2*t*y**3*z**3+t*sym.cos(t*x*y*z)))
    ) == 0
    assert og.Calc(CovariantD("a") @ var1("b")).get_components() == sym.Array(
        [
            [x*y*z, x**2*y**3*z**4, x*y*z*sym.cos(t*x*y*z), sym.exp(t+x+y+z)],
            [t*y*z, 2*t*x*y**3*z**4, t*y*z*sym.cos(t*x*y*z), sym.exp(t+x+y+z)],
            [t*x*z, 3*t*x**2*y**2*z**4, t*x*z*sym.cos(t*x*y*z), sym.exp(t+x+y+z)],
            [t*x*y, 4*t*x**2*y**3*z**3, t*x*y*sym.cos(t*x*y*z), sym.exp(t+x+y+z)]
        ]
    )
    assert og.Calc(CovariantD("a") @ var1("b")).get_components() == og.Calc(var1("b") @ CovariantD("a")).get_components()
    assert og.Calc(CovariantD("a") @ var1("a")).get_components() == og.Calc(var1("a") @ CovariantD("a")).get_components()
    assert sym.simplify(
        og.Calc(CovariantD("a") @ var2("a")).get_components()[()] -
        (t*(3 + sym.cos(r)*sym.cot(theta)))
    ) == 0


def test_christoffel():
    """
    Tests PyOGRe.Calc.CalcChristoffel() and Christoffel Module
    """

    t, x, r, theta, phi, M, k = sym.symbols("t x r theta phi M k", seq=True)
    a = sym.Function("a")(t)
    mc = minkowski.calc_christoffel().get_components()
    sc = schwarzschild.calc_christoffel().get_components()
    fc = flrw.calc_christoffel().get_components()

    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                assert mc[i1, i2, i3] == 0

    assert sc[0, 0, 1] == sc[0, 1, 0] == M/(r*(-2*M+r))
    assert sc[1, 2, 2] == 2*M - r
    assert sc[2, 1, 2] == sc[2, 2, 1] == sc[3, 1, 3] == sc[3, 3, 1] == 1/r
    assert fc[2, 1, 2] == fc[2, 2, 1] == fc[3, 1, 3] == fc[3, 3, 1] == 1/r
    assert fc[3, 2, 3] == fc[3, 3, 2] == 1/sym.tan(theta)
    assert sym.simplify(fc[0, 1, 1] - a * sym.diff(a, t) / (1 - k*r**2)) == 0

    simple_metric = cartesian.new_metric(
        name="Simple Metric",
        components=sym.Array(
            [
                [-x, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        ),
    )
    smc = simple_metric.calc_christoffel().get_components(coords=spherical)
    assert smc[0, 0, 1] == smc[0, 1, 0] == 1 / (2*r)
    assert smc[0, 0, 2] == smc[0, 2, 0] == 1 / (2 * sym.tan(theta))
    assert smc[0, 0, 3] == smc[0, 3, 0] == -sym.tan(phi) / 2
    assert smc[1, 0, 0] == sym.cos(phi)*sym.sin(theta) / 2
    assert smc[1, 2, 2] == -r
    assert smc[1, 3, 3] == -r*sym.sin(theta)**2
    assert smc[2, 0, 0] == sym.cos(phi)*sym.cos(theta) / (2*r)
    assert smc[2, 1, 2] == smc[2, 2, 1] == smc[3, 1, 3] == smc[3, 3, 1] == 1/r
    assert smc[2, 3, 3] == -sym.sin(2*theta) / 2
    assert smc[3, 0, 0] == -sym.sin(phi) / (2*r*sym.sin(theta))
    assert smc[3, 2, 3] == smc[3, 3, 2] == 1 / sym.tan(theta)


def test_riemann_tensor():
    """
    Tests PyOGRe.Calc.CalcRiemannTensor()
    """

    t, r, M = sym.symbols("t r M", seq=True)
    a = sym.Function("a")(t)
    mr = minkowski.calc_riemann_tensor().get_components()
    sr = schwarzschild.calc_riemann_tensor().get_components()
    fr = flrw.calc_riemann_tensor().get_components()

    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    assert mr[i1, i2, i3, i4] == 0

    assert sym.simplify(
        sr[0, 1, 0, 1] - 2*M/(r**2*(-2*M+r))
    ) == 0
    assert sym.simplify(
        -sr[0, 1, 1, 0] - 2*M/(r**2*(-2*M+r))
    ) == 0
    assert sr[0, 2, 0, 2] == sr[1, 2, 1, 2] == -sr[0, 2, 2, 0] == -sr[1, 2, 2, 1] == -M/r
    assert sr[3, 2, 2, 3] == -sr[3, 2, 3, 2] == -2*M/r

    assert fr[1, 3, 1, 3] == fr[2, 3, 2, 3] == sym.simplify(-fr[1, 3, 3, 1]) == sym.simplify(-fr[2, 3, 3, 2])
    assert fr[1, 0, 0, 1] == fr[2, 0, 0, 2] == fr[3, 0, 0, 3] == -fr[1, 0, 1, 0] == -fr[2, 0, 2, 0] == -fr[3, 0, 3, 0] == sym.diff(a, t, t) / a


def test_ricci_tensor():
    """
    Tests PyOGRe.Calc.CalcRicciTensor()
    """

    t, r, theta, k = sym.symbols("t r theta k", seq=True)
    a = sym.Function("a")(t)
    mr = minkowski.calc_ricci_tensor().get_components()
    sr = schwarzschild.calc_ricci_tensor().get_components()
    fr = flrw.calc_ricci_tensor().get_components()

    for i1 in range(4):
        for i2 in range(4):
            assert mr[i1, i2] == 0
            assert sr[i1, i2] == 0

    assert sym.simplify(
        fr[0, 0] + 3*sym.diff(a, t, t) / a
    ) == 0

    assert sym.simplify(
        fr[3, 3] - r**2*sym.sin(theta)**2 * (2*(k + sym.diff(a, t)**2) + a*sym.diff(a, t, t))
    ) == 0


def test_ricci_scalar():
    """
    Tests PyOGRe.Calc.CalcRicciScalar()
    """

    t, k = sym.symbols("t k", seq=True)
    a = sym.Function("a")(t)
    mr = minkowski.calc_ricci_scalar().get_components()[()]
    sr = schwarzschild.calc_ricci_scalar().get_components()[()]
    fr = flrw.calc_ricci_scalar().get_components()[()]

    assert mr == sr == 0

    assert sym.simplify(
        fr -
        6 * (k + sym.diff(a, t)**2 + a*sym.diff(a, t, t)) / a**2
    ) == 0


def test_weyl_tensor():
    """
    Tests PyOGRe.Calc.CalcWeylTensor()
    """

    r, theta, M = sym.symbols("r theta M", seq=True)
    mw = minkowski.calc_weyl_tensor().get_components()
    sw = schwarzschild.calc_weyl_tensor().get_components()
    fw = flrw.calc_weyl_tensor().get_components()
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    assert mw[i1, i2, i3, i4] == fw[i1, i2, i3, i4] == 0

    assert sw[2, 3, 2, 3] == -sw[2, 3, 3, 2] == -sw[3, 2, 2, 3] == sw[3, 2, 3, 2] == 2*M*r*sym.sin(theta)**2
    assert sw[0, 1, 0, 1] == -sw[0, 1, 1, 0] == -sw[1, 0, 0, 1] == sw[1, 0, 1, 0] == -2*M/r**3

    assert og.Calc(
        schwarzschild.calc_weyl_tensor()("a b c d") @ schwarzschild.calc_weyl_tensor()("a b c d")
    ).get_components()[()] == 48*M**2/r**6


def test_einstein():
    """
    Tests PyOGRe.Calc.CalcEinsteinTensor()
    """

    t, r, theta, k = sym.symbols("t r theta k", seq=True)
    a = sym.Function("a")(t)
    me = minkowski.calc_einstein_tensor().get_components()
    se = schwarzschild.calc_einstein_tensor().get_components()
    fe = flrw.calc_einstein_tensor().get_components()

    for i1 in range(4):
        for i2 in range(4):
            assert me[i1, i2] == se[i1, i2] == 0

    assert sym.simplify(
        fe[0, 0] - 3*(k + sym.diff(a, t)**2) / a**2
    ) == 0
    assert sym.simplify(
        fe[1, 1] -
        (k + sym.diff(a, t)**2 + 2*a*sym.diff(a, t, t)) / (-1 + k*r**2)
    ) == 0
    assert sym.simplify(
        fe[2, 2] +
        r**2*(k + sym.diff(a, t)**2 + 2*a*sym.diff(a, t, t))
    ) == 0
    assert sym.simplify(
        fe[3, 3] +
        r**2*sym.sin(theta)**2*(k + sym.diff(a, t)**2 + 2*a*sym.diff(a, t, t))
    ) == 0


def test_lagrangian():
    """
    Tests PyOGRe.Calc.CalcLagrangian() and Lagrangian Module
    """

    l, M, k = sym.symbols("lambda M k", seq=True)
    t, r, theta = sym.Function("t")(l), sym.Function("r")(l), sym.Function("theta")(l)
    a = sym.Function("a")(t)
    tdot, xdot, ydot, zdot = sym.Function("tdot")(l), sym.Function("xdot")(l), sym.Function("ydot")(l), sym.Function("zdot")(l)
    rdot, thetadot, phidot = sym.Function("rdot")(l), sym.Function("thetadot")(l), sym.Function("phidot")(l)
    ml = minkowski.calc_lagrangian().get_components()[()]
    sl = schwarzschild.calc_lagrangian().get_components()[()]
    fl = flrw.calc_lagrangian().get_components()[()]

    assert ml == -tdot**2 + xdot**2 + ydot**2 + zdot**2

    assert sym.simplify(
        sl - (
            rdot**2/(1-2*M/r) + (-1 + 2*M/r)*tdot**2 + r**2*(thetadot**2 + sym.sin(theta)**2*phidot**2)
        )
    ) == 0

    assert sym.simplify(
        fl - (
            -tdot**2 + a**2*(rdot**2/(1 - k*r**2) + r**2*(thetadot**2 + sym.sin(theta)**2*phidot**2))
        )
    ) == 0

    assert minkowski.calc_lagrangian(coords=spherical).get_components()[()] == minkowski.calc_lagrangian().get_components(coords=spherical)[()]

    x, y, z = sym.symbols("x y z", seq=True)
    mcl = minkowski.calc_lagrangian()

    assert mcl.show() is None
    assert mcl.list() is None

    assert mcl.list(replace={t: 0, x: 0, y: 0, z: 0}, function=abs) is None
    assert mcl.show(replace={t: 0, x: 0, y: 0, z: 0}, function=abs) is None
    assert mcl.get_components(replace={t: 0, x: 0, y: 0, z: 0}, function=abs) == sym.Array(0)


def test_lagrangian_geodesic():
    """
    Tests PyOGRe.Calc.CalcGeodesicFromLagrangian() and Lagrangian Module
    """

    l, M, k = sym.symbols("lambda M k", seq=True)
    t, r, theta, phi = sym.Function("t")(l), sym.Function("r")(l), sym.Function("theta")(l), sym.Function("phi")(l)
    a = sym.Function("a")(t)
    tdot = sym.Function("tdot")(l)
    rdot = sym.Function("rdot")(l)
    tddot, xddot, yddot, zddot = sym.Function("tddot")(l), sym.Function("xddot")(l), sym.Function("yddot")(l), sym.Function("zddot")(l)
    rddot = sym.Function("rddot")(l)
    mg = minkowski.calc_geodesic_from_lagrangian().get_components()
    sg = schwarzschild.calc_geodesic_from_lagrangian().get_components(replace={
        M: 1,
        theta: 0,
        phi: 0
    })
    fg = flrw.calc_geodesic_from_lagrangian().get_components(replace={
        k: 0,
        theta: 0,
        phi: 0
    })

    assert mg[0] == -tddot
    assert mg[1] == xddot
    assert mg[2] == yddot
    assert mg[3] == zddot

    assert sg[2] == sg[3] == 0
    assert sg[0] == -tddot + 2*tddot/r - 2*rdot*tdot/r**2
    assert sg[1] == r**4*rddot - 2*r**3*rddot - r**2*rdot**2 + r**2*tdot**2 - 4*r*tdot**2 + 4*tdot**2
    assert fg[2] == fg[3] == 0
    assert fg[0] == rdot**2*a*sym.diff(a, t) + tddot
    assert fg[1] == (rddot*a + 2*rdot*tdot*sym.diff(a, t))*a

    assert minkowski.calc_geodesic_from_lagrangian(coords=spherical).get_components() == minkowski.calc_geodesic_from_lagrangian().get_components(coords=spherical)


def test_christoffel_geodesic():
    """
    Tests PyOGRe.Calc.CalcGeodesicFromChristoffel() and Lagrangian Module
    """

    l, M, k = sym.symbols("lambda M k", seq=True)
    t, r, theta, phi = sym.Function("t")(l), sym.Function("r")(l), sym.Function("theta")(l), sym.Function("phi")(l)
    a = sym.Function("a")(t)
    tdot = sym.Function("tdot")(l)
    rdot = sym.Function("rdot")(l)
    tddot, xddot, yddot, zddot = sym.Function("tddot")(l), sym.Function("xddot")(l), sym.Function("yddot")(l), sym.Function("zddot")(l)
    rddot = sym.Function("rddot")(l)
    mg = minkowski.calc_geodesic_from_christoffel().get_components()
    sg = schwarzschild.calc_geodesic_from_christoffel().get_components(replace={
        M: 1,
        theta: 0,
        phi: 0
    })
    fg = flrw.calc_geodesic_from_christoffel().get_components(replace={
        k: 0,
        theta: 0,
        phi: 0
    })

    assert mg[0] == tddot
    assert mg[1] == xddot
    assert mg[2] == yddot
    assert mg[3] == zddot

    assert sg[2] == sg[3] == 0
    assert sg[0] == r**2*tddot - 2*r*tddot + 2*rdot*tdot
    assert sg[1] == r**4*rddot - 2*r**3*rddot - r**2*rdot**2 + r**2*tdot**2 - 4*r*tdot**2 + 4*tdot**2

    assert fg[2] == fg[3] == 0
    assert fg[0] == -rdot**2*a*sym.diff(a, t) - tddot
    assert fg[1] == -rddot*a - 2*rdot*tdot*sym.diff(a, t)

    assert minkowski.calc_geodesic_from_christoffel(coords=spherical).get_components() == minkowski.calc_geodesic_from_christoffel().get_components(coords=spherical)


def test_time_geodesic():
    """
    Tests PyOGRe.Calc.CalcGeodesicWithTimeParameter() and Lagrangian Module
    """

    t, x, y, z, M, k = sym.symbols("t x y z M k", seq=True)
    r, theta, phi = sym.Function("r")(t), sym.Function("theta")(t), sym.Function("phi")(t)
    a = sym.Function("a")(t)
    rdot = sym.Function("rdot")(t)
    xddot, yddot, zddot = sym.Function("xddot")(t), sym.Function("yddot")(t), sym.Function("zddot")(t)
    rddot = sym.Function("rddot")(t)
    mg = minkowski.calc_geodesic_with_time_parameter().get_components()
    sg = schwarzschild.calc_geodesic_with_time_parameter().get_components(replace={
        M: 1,
        theta: 0,
        phi: 0
    })
    fg = flrw.calc_geodesic_with_time_parameter().get_components(replace={
        k: 0,
        theta: 0,
        phi: 0
    })

    assert mg[0] == 0
    assert mg[1] == xddot
    assert mg[2] == yddot
    assert mg[3] == zddot

    assert sg[0] == sg[2] == sg[3] == 0
    assert sg[1] == r**4*rddot - 2*r**3*rddot - 3*r**2*rdot**2 + r**2 - 4*r + 4

    assert fg[0] == fg[2] == fg[3] == 0
    assert fg[1] == -rddot*a + rdot**3*a**2*sym.diff(a, t) - 2*rdot*sym.diff(a, t)

    assert minkowski.calc_geodesic_with_time_parameter(coords=spherical).get_components() == minkowski.calc_geodesic_with_time_parameter().get_components(coords=spherical)

    x, y, z = sym.symbols("x y z", seq=True)
    mgt = minkowski.calc_geodesic_with_time_parameter()

    assert mgt.show() is None
    assert mgt.list() is None

    assert mgt.list(replace={x: 0, y: 0, z: 0}, function=abs) is None
    assert mgt.show(replace={x: 0, y: 0, z: 0}, function=abs) is None
    assert mgt.get_components(replace={x: 0, y: 0, z: 0}, function=abs) == sym.Array([0, 0, 0, 0])


def test_calc_function():
    """
    Tests PyOGRe.Calc.Calc() function
    """

    with pytest.raises(TypeError):
        og.Calc([1, 2])

    with pytest.raises(TypeError):
        og.Calc(minkowski("a b"), indices=42)

    with pytest.raises(ValueError):
        og.Calc(minkowski("a b"), indices="b c")
