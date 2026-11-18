"""Make the logo-bundle files of a tool: menu icon, glyphs, app icon, horizontal logos, PNG, ICO and PDF.

    python tools/make_tool_logos.py                 # every tool defined below
    python tools/make_tool_logos.py pyDENsity       # one tool

Writes into ``docs/pySASfit-logo-bundle`` the same set of files the other tools have (see its README.txt):

    menu-icons/<tool>/<tool>-menu.svg, <tool>-menu.ico, png/<tool>-menu-<n>.png (16 ... 256),
                      glyph/<tool>-glyph-black|white.svg, glyph/png/<tool>-glyph-black|white-<n>.png (16, 24, 32)
    icons/<tool>/<tool>-icon.svg, <tool>-icon-small.svg, <tool>.ico, png/<tool>-icon-<n>.png (16 ... 1024)
    logos/<tool>/svg/<tool>-light|dark|mono-black|mono-white.svg, png/<tool>-<variant>-h256|h1024.png
    print/<tool>/<tool>-light.pdf, <tool>-mono-black.pdf

Afterwards ``python tools/sync_logos.py`` copies what the programs use into ``src/pysasfit/resources``.

The wordmark is written in Space Grotesk (SIL Open Font License) and converted to outlines exactly like the
existing logos: size 120, baseline 142, letter spacing -0.023 em, "py" and the ending in Regular (400), the core
in Bold (700), starting at x = 240 next to the tile (scaled to 200 px). The font files are not part of pySASfit:
pass ``--font-dir`` with ``space-grotesk-latin-400-normal.woff`` and ``...-700-normal.woff`` (e.g. from the npm
package @fontsource/space-grotesk) or the static TTFs (SpaceGrotesk-Regular.ttf, SpaceGrotesk-Bold.ttf).

Needs: fonttools (+ brotli for woff2), cairosvg, Pillow.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "docs", "pySASfit-logo-bundle")

NAVY, NAVY_DARK, CYAN, GOLD, GOLD_APP = "#0E2740", "#153A5E", "#7FD3E8", "#FFC857", "#FFB547"
GREY, GREY_DARK = "#5E7388", "#9FB6CC"


# ===================================================================================== glyphs (120 x 120 units)
def _poly(points, close=True):
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in points)
    return d + (" Z" if close else "")


def _lobes(lev, n=96):
    """Contour of a two-lobed density (pyDENsity), scaled by ``lev``."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        r = lev * (0.30 + 0.07 * math.cos(2 * t) + 0.035 * math.cos(3 * t + 0.6))
        pts.append((120 * (0.5 + r * math.cos(t) * 1.12), 120 * (0.52 + r * math.sin(t) * 0.92)))
    return _poly(pts)


def _blob(cx, cy, r, n=80):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        rr = r * (1 + 0.08 * math.cos(3 * t + cx * 7))
        pts.append((120 * (cx + rr * math.cos(t)), 120 * (cy + rr * math.sin(t))))
    return _poly(pts)


def glyph_density(fg, accent, tile, w=1.0):
    """Nested contour lines of a two-lobed electron density, the innermost filled."""
    return (f'<path d="{_lobes(1.0)}" fill="none" stroke="{fg}" stroke-width="{7 * w:.1f}" stroke-linejoin="round"/>'
            f'<path d="{_lobes(0.66)}" fill="none" stroke="{fg}" stroke-width="{6 * w:.1f}" stroke-linejoin="round"/>'
            f'<path d="{_lobes(0.34)}" fill="{accent}"/>')


def glyph_multi(fg, accent, tile, w=1.0):
    """Two overlapping lobes (two components of a multi-contrast map), the first with an inner contour."""
    if tile == "none":       # one-colour glyph: the first lobe as an outline, the second filled
        return (f'<path d="{_blob(0.40, 0.56, 0.27)}" fill="none" stroke="{fg}" stroke-width="{7 * w:.1f}"/>'
                f'<path d="{_blob(0.65, 0.40, 0.21)}" fill="{fg}"/>'
                f'<ellipse cx="45.6" cy="68.4" rx="14" ry="13" fill="none" stroke="{fg}" stroke-width="{5 * w:.1f}"/>')
    return (f'<path d="{_blob(0.40, 0.56, 0.27)}" fill="{fg}" stroke="{tile}" stroke-width="{4 * w:.1f}"/>'
            f'<path d="{_blob(0.65, 0.40, 0.21)}" fill="{accent}" stroke="{tile}" stroke-width="{4 * w:.1f}"/>'
            f'<ellipse cx="45.6" cy="68.4" rx="14" ry="13" fill="none" stroke="{tile}" stroke-width="{5 * w:.1f}"/>')


def glyph_scout(fg, accent, tile, w=1.0):
    """A scattering curve (plateau, fall) with a magnifier on its knee: the search for the model."""
    pts = []
    for i in range(41):
        x = i / 40
        y = 1.0 / (1.0 + (2.6 * x) ** 4) + 0.05 * (1 - x)
        pts.append((120 * (0.12 + 0.76 * x), 120 * (0.80 - 0.56 * min(y, 1.0))))
    c, r = (67.2, 50.4), 20.4
    hx0, hy0 = c[0] + 0.72 * r, c[1] + 0.72 * r
    if accent == fg:         # one colour: an open lens
        accent = "none"
    return (f'<path d="{_poly(pts, close=False)}" fill="none" stroke="{fg}" stroke-width="{7 * w:.1f}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{c[0]}" cy="{c[1]}" r="{r}" fill="{accent}" stroke="{fg}" '
            f'stroke-width="{8 * w:.1f}"/>'
            f'<path d="M{hx0:.2f} {hy0:.2f} L103.2 86.4" fill="none" stroke="{fg}" stroke-width="{12 * w:.1f}" '
            f'stroke-linecap="round"/>')


def glyph_atsas(fg, accent, tile, w=1.0):
    """A distance distribution p(r) (bell with a tail, the area under it lightly filled) and a small bead model
    above its tail: the two main products of the ATSAS programs (GNOM, DAMMIF)."""
    f = [(i / 48) ** 1.6 * math.exp(-(((i / 48) / 0.40) ** 2.2)) for i in range(49)]
    top = max(f)
    pts = [(120 * (0.12 + 0.76 * i / 48), 120 * (0.80 - 0.56 * v / top)) for i, v in enumerate(f)]
    line = _poly(pts, close=False)
    base = 120 * 0.80
    area = line + f" L{pts[-1][0]:.2f} {base:.2f} L{pts[0][0]:.2f} {base:.2f} Z"
    beads = [(80, 26), (93, 26), (86.5, 15), (99.5, 15), (86.5, 37), (99.5, 37)]
    r = 6.6 * (1 + 0.12 * (w - 1))
    out = []
    if tile != "none" and accent != fg:
        out.append(f'<path d="{area}" fill="{fg}" fill-opacity=".25"/>')
    out.append(f'<path d="{line}" fill="none" stroke="{fg}" stroke-width="{7.5 * w:.1f}" stroke-linecap="round" '
               f'stroke-linejoin="round"/>')
    out += [f'<circle cx="{x}" cy="{y}" r="{r:.2f}" fill="{accent}"/>' for x, y in beads]
    return "".join(out)


#: tool -> (wordmark parts, tile colour, dark-theme core colour, glyph function, one-line description)
TOOLS = {
    "pyATsas": (("py", "AT", "sas"), "#0B6666", "#5FA8A8", glyph_atsas,
                "p(r) curve of GNOM with a small DAMMIF bead model in gold"),
    "pyDENsity": (("py", "DEN", "sity"), "#2B4C9B", "#8FA6D9", glyph_density,
                  "nested contours of a two-lobed electron density, innermost in gold"),
    "pyDENSSmulti": (("py", "DENSS", "multi"), "#1F7A8C", "#6FB3C0", glyph_multi,
                     "two overlapping lobes (white, gold): components of a multi-contrast map"),
    "pySASscout": (("py", "SAS", "scout"), "#9A6A00", "#D2A84A", glyph_scout,
                   "scattering curve with a magnifier on its knee"),
}


def _svg(body, title, vb="0 0 120 120", size=True):
    wh = ' width="120" height="120"' if size else ""
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}"{wh}><title>{title}</title>{body}</svg>'


def menu_svg(tool, tile, glyph):
    return _svg(f'<rect width="120" height="120" rx="26" fill="{tile}"/>' + glyph("#FFFFFF", GOLD, tile), tool)


def glyph_svg(tool, colour, glyph):
    # one colour, transparent background: the tile-coloured separating strokes are left out (drawn in "none")
    return _svg(glyph(colour, colour, "none"), f"{tool} glyph")


def app_icon_svg(tool, glyph, small=False):
    """The navy family tile of icons/ (cyan glyph, amber accent)."""
    w = 1.4 if small else 1.0
    return _svg(f'<rect width="120" height="120" rx="28" fill="{NAVY}"/>' + glyph(CYAN, GOLD_APP, NAVY, w), tool)


# ===================================================================================== wordmark
class Wordmark:
    def __init__(self, font_dir):
        from fontTools.ttLib import TTFont
        self.fonts = {}
        for w, names in ((400, ("space-grotesk-latin-400-normal.woff", "space-grotesk-latin-400-normal.woff2",
                                "SpaceGrotesk-Regular.ttf")),
                         (700, ("space-grotesk-latin-700-normal.woff", "space-grotesk-latin-700-normal.woff2",
                                "SpaceGrotesk-Bold.ttf"))):
            for n in names:
                p = os.path.join(font_dir, n)
                if os.path.isfile(p):
                    self.fonts[w] = TTFont(p)
                    break
            else:
                raise SystemExit(f"Space Grotesk {w} not found in {font_dir} (one of {names})")

    def paths(self, parts, size=120.0, base=142.0, x=240.0, track=-2.76):
        """[(path d, part index)] of "py", core, ending, and the x where the text ends."""
        from fontTools.pens.svgPathPen import SVGPathPen
        from fontTools.pens.transformPen import TransformPen
        out = []
        sc = size / 1000.0
        first = True
        for k, (txt, w) in enumerate(zip(parts, (400, 700, 400))):
            f = self.fonts[w]
            gs, cmap, hm = f.getGlyphSet(), f.getBestCmap(), f["hmtx"]
            cmds = []
            for ch in txt:
                if not first:
                    x += track
                first = False
                g = cmap[ord(ch)]
                pen = SVGPathPen(gs, ntos=lambda v: f"{v:.2f}")
                gs[g].draw(TransformPen(pen, (sc, 0, 0, -sc, x, base)))
                cmds.append(pen.getCommands())
                x += hm[g][0] * sc
            out.append(" ".join(cmds))
        return out, x


def logo_svg(tool, parts, tile, core_dark, glyph, wm, variant):
    (p_py, p_core, p_end), end = wm.paths(parts)
    width = int(math.ceil(end)) + 10
    if variant == "light":
        tile_c, fg, acc, cols = tile, "#FFFFFF", GOLD, (GREY, tile, GREY)
    elif variant == "dark":
        tile_c, fg, acc, cols = tile, "#FFFFFF", GOLD, (GREY_DARK, core_dark, GREY_DARK)
    elif variant == "mono-black":       # glyph in one colour on the tile (accent = glyph colour)
        tile_c, fg, acc, cols = "#000000", "#FFFFFF", "#FFFFFF", ("#000000",) * 3
    else:                                                        # mono-white
        tile_c, fg, acc, cols = "#FFFFFF", "#000000", "#000000", ("#FFFFFF",) * 3
    body = (f'<g transform="scale(1.66667)"><rect width="120" height="120" rx="26" fill="{tile_c}"/>'
            + glyph(fg, acc, tile_c) + "</g>"
            + "".join(f'<path d="{d}" fill="{c}"/>' for d, c in zip((p_py, p_core, p_end), cols)))
    return _svg(body, tool, vb=f"-10 -10 {width} 220", size=False)


# ===================================================================================== raster / ico / pdf
def _png(svg, out, width=None, height=None):
    import cairosvg
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=out, output_width=width, output_height=height)


def _ico(svgs_by_size, out):
    """PNG-encoded Windows icon from {size: svg}."""
    import cairosvg
    from PIL import Image
    ims = []
    for s, svg in sorted(svgs_by_size.items()):
        ims.append(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode(), output_width=s,
                                                          output_height=s))).convert("RGBA"))
    big = ims[-1]
    big.save(out, format="ICO", sizes=[(im.width, im.height) for im in ims], append_images=ims[:-1])


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def make(tool, wm, bundle=BUNDLE):
    import cairosvg
    parts, tile, core_dark, glyph, _desc = TOOLS[tool]
    written = []

    def w(rel, text):
        p = os.path.join(bundle, rel)
        _write(p, text)
        written.append(p)

    # menu icons
    m = menu_svg(tool, tile, glyph)
    w(f"menu-icons/{tool}/{tool}-menu.svg", m)
    for s in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        _png(m, os.path.join(bundle, f"menu-icons/{tool}/png/{tool}-menu-{s}.png"), s, s)
    _ico({s: m for s in (16, 24, 32, 48, 64, 128, 256)}, os.path.join(bundle, f"menu-icons/{tool}/{tool}-menu.ico"))
    for name, col in (("black", "#000000"), ("white", "#FFFFFF")):
        g = glyph_svg(tool, col, glyph)
        w(f"menu-icons/{tool}/glyph/{tool}-glyph-{name}.svg", g)
        for s in (16, 24, 32):
            _png(g, os.path.join(bundle, f"menu-icons/{tool}/glyph/png/{tool}-glyph-{name}-{s}.png"), s, s)
    # app icons (navy family)
    a, a_small = app_icon_svg(tool, glyph), app_icon_svg(tool, glyph, small=True)
    w(f"icons/{tool}/{tool}-icon.svg", a)
    w(f"icons/{tool}/{tool}-icon-small.svg", a_small)
    for s in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        _png(a_small if s <= 32 else a, os.path.join(bundle, f"icons/{tool}/png/{tool}-icon-{s}.png"), s, s)
    _ico({s: (a_small if s <= 32 else a) for s in (16, 24, 32, 48, 64, 128, 256)},
         os.path.join(bundle, f"icons/{tool}/{tool}.ico"))
    # horizontal logos
    for v in ("light", "dark", "mono-black", "mono-white"):
        lg = logo_svg(tool, parts, tile, core_dark, glyph, wm, v)
        w(f"logos/{tool}/svg/{tool}-{v}.svg", lg)
        for h in (256, 1024):
            _png(lg, os.path.join(bundle, f"logos/{tool}/png/{tool}-{v}-h{h}.png"), height=h)
        if v in ("light", "mono-black"):
            p = os.path.join(bundle, f"print/{tool}/{tool}-{v}.pdf")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cairosvg.svg2pdf(bytestring=lg.encode(), write_to=p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tools", nargs="*", default=list(TOOLS))
    ap.add_argument("--font-dir", required=True, help="folder with the Space Grotesk 400 and 700 font files")
    ap.add_argument("--bundle", default=BUNDLE)
    a = ap.parse_args(argv)
    wm = Wordmark(a.font_dir)
    for t in a.tools:
        make(t, wm, a.bundle)
        print(f"{t}: written to {a.bundle}")


if __name__ == "__main__":
    sys.exit(main())
