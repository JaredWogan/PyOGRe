import sympy as sym

from PyOGRe.Coordinates import new_coordinates

t, x, y, z, r, theta, phi, M, k = sym.symbols("t x y z r theta phi M k")
a = sym.Function("a")(t)

cartesian = new_coordinates(
    name="4D Cartesian",
    components=sym.Array([t, x, y, z])
)

spherical = new_coordinates(
    name="4D Spherical",
    components=sym.Array([t, r, theta, phi])
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
        None,
        sym.Eq(r, sym.sqrt(x**2+y**2+z**2)),
        sym.Eq(theta, sym.acos(z/(sym.sqrt(x**2+y**2+z**2)))),
        sym.Eq(phi, sym.atan(y/x)),
    ]
)

minkowski = cartesian.new_metric(
    name="4D Minkowski",
    components=sym.diag(-1, 1, 1, 1)
)

schwarzschild = spherical.new_metric(
    name="Schwarzschild",
    components=sym.Array(
        [
            [-(1-2*M/r), 0, 0, 0],
            [0, 1/(1-2*M/r), 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sym.sin(theta)**2]
        ]
    )
)

flrw = spherical.new_metric(
    name="FLRW",
    components=sym.Array(
        [
            [-1, 0, 0, 0],
            [0, a**2/(1-k*r**2), 0, 0],
            [0, 0, a**2*r**2, 0],
            [0, 0, 0, a**2*r**2*sym.sin(theta)**2]
        ]
    )
)
