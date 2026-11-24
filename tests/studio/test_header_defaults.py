"""Default values and domains in the parameter tables of the plugin headers (syntax of 2026-11-24)."""

from __future__ import annotations

import os
import sys

import pytest

from pysasfit.studio import headers

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEADER = """/* ################ start ff_demo ################ */
/**
 * \\defgroup ff_demo demo
 * \\ingroup ff_plugins_spheres_shells
 *
 * \\brief demo function
 *
 * \\par Required parameters:
 *      <table border="0"><tr>
 *       <td>\\b R_c</td>
 *       <td>core radius</td>
 *       <td>(default=10)</td>
 *       <td>[0;+inf)</td>
 *      </tr><tr>
 *       <td>\\b eta</td>
 *       <td>(default=1e10)</td>
 *       <td>(-inf;+inf)</td>
 *       <td></td>
 *      </tr><tr>
 *       <td>\\b R2</td>
 *       <td>inner radius</td>
 *       <td>(default=5)</td>
 *       <td>[0;R_c]</td>
 *      </tr><tr>
 *       <td>\\b n</td>
 *       <td>number of layers</td>
 *      </tr></table>
 *
 * \\par Default values and domains:
 *      R_c = 20 [0, inf); eta = 1; R2 = 1; n = 3 [1, inf)
 */
"""


def test_cells_are_read_in_any_order_and_win_over_the_paragraph(tmp_path):
    p = tmp_path / "sasfit_demo.h"
    p.write_text(HEADER)
    (info,) = headers.parse_header(str(p))
    r, eta, r2, n = info.parameters
    assert (r.default, r.bounds, r.description) == (10.0, (0.0, float("inf")), "core radius")
    assert (eta.default, eta.bounds, eta.description) == (1e10, (float("-inf"), float("inf")), "")
    assert r2.domain == "[0;R_c]" and r2.bounds == (0.0, float("inf"))          # bound given by another parameter
    assert (n.default, n.bounds) == (3.0, (1.0, float("inf")))                  # older paragraph as fallback


def _writer():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import write_header_defaults

    return write_header_defaults


class _P:
    def __init__(self, name, default, lower=float("-inf"), upper=float("inf"), lo_open=False, hi_open=False):
        self.name, self.default, self.lower, self.upper = name, default, lower, upper
        self.lower_open, self.upper_open, self.kind = lo_open, hi_open, "x"


def test_writer_cells_paragraph_and_hand_values():
    w = _writer()
    text = HEADER.replace("\n", "\r\n")
    params = [_P("R_c", 20.0, 0.0), _P("eta", 1e10), _P("R2", 5.0, 0.0), _P("n", 3.0, 1.0)]
    new, status, notes = w.update_header(text, "ff_demo", params, "\r\n")
    assert status == "written"
    assert "Default values and domains" not in new
    assert new.count("\n") == new.count("\r\n")                                 # CRLF kept
    assert "<td>(default=10)</td>" in new and "R_c: header default 10 kept (port 20)" in notes[0]
    assert "<td>(default=1e10)</td>" in new                                     # hand-written text kept
    assert "<td>[0;R_c]</td>" in new                                            # symbolic domain kept
    assert "<td>\\b n</td>\r\n *       <td>number of layers</td>\r\n *       <td>(default=3)</td>\r\n" \
           " *       <td>[1;+inf)</td>" in new
    # eta row: name, (empty) description, default, domain
    i = new.index("<td>\\b eta</td>")
    assert new[i:].split("</tr>")[0].count("<td>") == 4
    forced, _, _ = w.update_header(text, "ff_demo", params, "\r\n", force=True)
    assert "<td>(default=20)</td>" in forced
    assert w._num(1e10) == "1e10" and w._num(0.001) == "0.001" and w._num(float("inf")) == "+inf"

