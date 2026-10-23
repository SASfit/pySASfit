"""Names of parameters in constraint expressions (no SASfit library needed)."""

import pytest

from pysasfit.studio import fitting


def par(label, address="0.sd.mu"):
    return fitting.FitParameter(label, address, (0,), 1.0, 0.0, 10.0, False)


@pytest.mark.parametrize(
    "label, ident",
    [
        ("sd.mu", "sd_mu"),
        ("ff.R", "ff_R"),
        ("background", "background"),
        ("1.sd.mu", "c1_sd_mu"),  # starts with a digit: not a valid Python name without the prefix
        ("scale[a b]", "scale_a_b"),
    ],
)
def test_identifier_of_a_label(label, ident):
    assert fitting._ident(label) == ident
    assert par(label).ident == ident
    assert ident.isidentifier()


def test_constraint_with_a_second_contribution_parses():
    c = fitting._Constraint("sd_mu > c1_sd_mu")
    assert c.names == {"sd_mu", "c1_sd_mu"}


def test_a_digit_label_does_not_clash_with_the_plain_one():
    assert fitting._ident("1.x") != fitting._ident("x")
