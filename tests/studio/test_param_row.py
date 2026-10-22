"""ParameterRow (one row of the model explorer): slider range, editing, bounds, fit box. No SASfit library needed."""

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.studio.param_row import ParameterRow, clear_layout, slider_range  # noqa: E402

INF = math.inf


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ---- slider_range (pure function) ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, bounds, expected",
    [
        (5.0, (-INF, INF), (0.0, 10.0)),
        (-4.0, (-INF, INF), (-8.0, 0.0)),
        (0.0, (-INF, INF), (0.0, 1.0)),
        (5.0, (1.0, 8.0), (1.0, 8.0)),  # clipped to the bounds
        (5.0, (6.0, INF), (5.0, 10.0)),  # the value stays inside, even if outside the bounds
    ],
)
def test_slider_range(value, bounds, expected):
    assert slider_range(value, bounds) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-3.0, 0.0, 1e-9, 7.5, 1e6])
def test_slider_range_always_contains_the_value_and_is_not_empty(value):
    a, b = slider_range(value)
    assert a <= value <= b and b > a


def test_slider_range_of_a_fixed_parameter_is_not_degenerate():
    a, b = slider_range(2.0, (2.0, 2.0))
    assert b > a and a <= 2.0 <= b


# ---- the widget ---------------------------------------------------------------------------------------------


def test_the_value_is_shown_in_the_edit_field(app):
    row = ParameterRow("R", 12.5)
    assert row.edit.text() == "12.5"
    assert row.value == 12.5


def test_typing_a_value_emits_and_recentres_the_slider(app):
    row = ParameterRow("R", 10.0, sliders=True)
    got = []
    row.valueChanged.connect(got.append)
    row.edit.setText("40")
    row.edit.editingFinished.emit()
    assert got == [40.0]
    assert (row.a, row.b) == pytest.approx((0.0, 80.0))


def test_a_decimal_comma_is_accepted(app):
    row = ParameterRow("R", 1.0)
    row.edit.setText("2,5")
    row.edit.editingFinished.emit()
    assert row.value == 2.5


@pytest.mark.parametrize("text", ["abc", "", "nan", "inf"])
def test_invalid_text_reverts_without_signal(app, text):
    row = ParameterRow("R", 3.0)
    got = []
    row.valueChanged.connect(got.append)
    row.edit.setText(text)
    row.edit.editingFinished.emit()
    assert got == []
    assert row.value == 3.0
    assert row.edit.text() == "3"


def test_moving_the_slider_changes_the_value(app):
    row = ParameterRow("R", 10.0, sliders=True)  # range 0..20
    got = []
    row.valueChanged.connect(got.append)
    row.slider.setValue(row.STEPS // 4)
    assert got and got[-1] == pytest.approx(5.0)
    row.slider.setValue(row.STEPS)
    assert row.value == pytest.approx(20.0)
    assert float(row.edit.text()) == pytest.approx(20.0)


def test_set_value_can_be_silent_and_keep_the_slider_range(app):
    row = ParameterRow("R", 10.0)
    got = []
    row.valueChanged.connect(got.append)
    row.set_value(30.0, emit=False, recenter=False)
    assert got == []
    assert row.value == 30.0
    assert (row.a, row.b) == pytest.approx((0.0, 20.0))
    assert row.slider.value() == row.STEPS  # clipped to the end of the range


def test_sliders_can_be_shown_and_hidden(app):
    row = ParameterRow("R", 1.0, sliders=False)
    row.show()
    assert not row.slider.isVisible()
    row.set_sliders(True)
    assert row.slider.isVisible()
    row.close()


def test_the_fit_box_shows_the_bounds_and_emits(app):
    row = ParameterRow("R", 1.0)
    row.show()
    got = []
    row.fitToggled.connect(got.append)
    assert not row.is_fit() and not row.bounds_box.isVisible()
    row.fit.setChecked(True)
    assert got == [True] and row.is_fit() and row.bounds_box.isVisible()
    row.close()


def test_set_fit_is_silent_unless_asked(app):
    row = ParameterRow("R", 1.0)
    got = []
    row.fitToggled.connect(got.append)
    row.set_fit(True)
    assert row.is_fit() and got == []
    row.set_fit(False, emit=True)
    assert not row.is_fit() and got == [False]


def test_bounds_are_taken_over_and_cleaned(app):
    row = ParameterRow("R", 1.0, bounds=(0.5, INF))
    assert row.bounds_text() == ("0.5", "")
    got = []
    row.boundsEdited.connect(lambda: got.append(1))
    row.upper.setText(" 3,5 ")
    row.upper.editingFinished.emit()
    assert row.bounds_text() == ("0.5", "3.5")
    assert got


def test_an_invalid_bound_reverts(app):
    row = ParameterRow("R", 1.0, bounds=(0.5, 2.0))
    row.lower.setText("oops")
    row.lower.editingFinished.emit()
    assert row.bounds_text() == ("0.5", "2")


def test_a_dummy_parameter_cannot_be_edited(app):
    row = ParameterRow("x", 1.0, dummy=True)
    assert not row.edit.isEnabled() and not row.fit.isEnabled()
    row.set_editable(True)  # a dummy stays disabled
    assert not row.edit.isEnabled()
    assert "dummy" in row.edit.toolTip()


def test_set_editable_disables_and_explains(app):
    row = ParameterRow("R", 1.0, description="radius")
    row.set_editable(False, "set by the contrast")
    assert not row.edit.isEnabled() and not row.slider.isEnabled()
    assert row.edit.toolTip() == "set by the contrast"
    row.set_editable(True)
    assert row.edit.isEnabled() and row.edit.toolTip() == "radius"


def test_the_distribution_radio_sits_in_front_of_the_fit_box(app):
    row = ParameterRow("R", 1.0)
    group = QtWidgets.QButtonGroup()
    radio = row.add_radio(group)
    assert row.radio is radio and radio in group.buttons()
    assert row._layout.indexOf(radio) == row._layout.indexOf(row.fit) - 1


def test_clear_layout_removes_the_widgets(app):
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    for i in range(3):
        lay.addWidget(QtWidgets.QLabel(str(i)))
    clear_layout(lay)
    assert lay.count() == 0
