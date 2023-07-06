import os
import pytest
import sympy as sym
import PyOGRe as og


def test_export():
    """
    Tests PyOGRe.Export
    """

    og.clear_instances()

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

    test_import_1 = r"""
    <|"Imported Cartesian"-><|"Components"-><|{{1},"4D Cartesian"}->{t, x, y, z}|>,"DefaultCoords"->
    "4D Cartesian","DefaultIndices"->{1},"Role"->"Coordinates","Symbol"->x,"CoordTransformations"-><|"Spherical"
    ->{x->r*Sin[theta]*Cos[phi],y->r*Sin[phi]*Sin[theta],z->r*Cos[theta]}|>,"OGReVersion"->"PyOGRe v0.0.1"|>|>
    """

    test_import_2 = r"""<|"Minkowski" -> <|"Components" -> <|{{-1, -1},
    "Cartesian"} -> {{-1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0,
    0, 0, 1}}, {{1, 1},
    "Cartesian"} -> {{-1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0,
    0, 0, 1}}, {{1, -1},
    "Cartesian"} -> {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0,
    0, 0, 1}}, {{-1, 1},
    "Cartesian"} -> {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0,
    0, 0, 1}}|>, "DefaultCoords" -> "Cartesian",
   "DefaultIndices" -> {-1, -1}, "Metric" -> "Minkowski",
   "Role" -> "Metric", "Symbol" -> "g",
   "OGReVersion" -> "v1.6 (2021-08-07)"|>|>
    """

    with pytest.raises(TypeError):
        og.export_all(42)

    test_1 = og.import_from_string(test_import_1)
    test_2 = og.import_from_string(test_import_2)

    assert test_1.show() is None
    assert test_2.list() is None

    assert minkowski.get_components() == test_2.get_components()

    assert og.import_from_string(minkowski.export()).get_components() == minkowski.get_components()

    assert cartesian.export() == r"""<|"Cartesian"-><|"Components"-><|{{1},"Cartesian"}->{t, x, y, z}\\ReleaseHold|>,"DefaultCoords"->"Cartesian","DefaultIndices"->{1}\\ReleaseHold,"Role"->"Coordinates","Symbol"->x\\ReleaseHold,"CoordTransformations"-><|"Spherical"->{x->r*Sin[theta]*Cos[phi]\\ReleaseHold,y->r*Sin[phi]*Sin[theta]\\ReleaseHold,z->r*Cos[theta]\\ReleaseHold}|>,"OGReVersion"->"PyOGRe v1.0.0"|>|>"""

    minkowski.export("temp.txt")
    assert og.import_from_file("temp.txt").get_components() == minkowski.get_components()
    os.remove("temp.txt")

    og.export_all("temp.txt")
    assert len(og.import_all_from_string(og.export_all())) == len(og.import_all_from_file("temp.txt"))
    os.remove("temp.txt")
