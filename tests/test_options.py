import pytest
import sympy as sym
import PyOGRe as og


def test_options():
    """
    Tests PyOGRe.Options
    """

    assert og.set_index_letters() is None
    with pytest.raises(TypeError):
        og.set_index_letters(1)
    assert og.set_index_letters("a b c d e f g") is None
    assert og.set_index_letters("automatic") is None

    assert og.set_curve_parameter() is None
    with pytest.raises(TypeError):
        og.set_curve_parameter(1)
    assert og.set_curve_parameter("a b c d e f g") is None
    assert og.get_curve_parameter() == sym.Symbol("a")
    assert og.set_curve_parameter("automatic") is None
    assert og.get_curve_parameter() == sym.Symbol("lambda")

    assert og.command_line_support() is None
    assert og.jupyter_support() is None

    assert og.get_options() is None

    assert isinstance(og.get_instances(), list)

    assert og.delete_results() is None
