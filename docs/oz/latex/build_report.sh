#!/bin/sh
# Build the combined ozGUI/ozLib report.
#
#   sh build_report.sh
#
# FOUR passes with bibtex between them. The sequence matters:
#   pdflatex   writes the .aux, recording which entries are cited
#   bibtex     reads the .aux and ozgui.bib, writes the .bbl
#   pdflatex   pulls the .bbl in
#   pdflatex   resolves citation labels, \part numbering and the table of
#              contents
# Skip the bibtex pass and every citation renders as (?) in an otherwise
# clean build, which is easy not to notice.
#
# STRUCTURE. The master is ozGUI_ozLib_report.tex. It pulls in
#   ozGUI_ozLib_documentation.tex             (Part I)
#   ozGUI_ozLib_multicomponent_supplement.tex (Part II)
# via \input, with docmute discarding their preambles, so both remain
# independently buildable on their own. The bibliography is the shared
# ozgui.bib, which the manuscript also uses -- add entries there, not to a
# thebibliography block in a fragment.
#
# TWO PACKAGES THIS NEEDS, and why:
#   lmodern    supplies TS1 in Type 1. Without it a TS1 glyph sends pdflatex
#              looking for a bitmap font it cannot build, and the run dies
#              with "Font tcrm1000 at 600 not found" and NO PDF.
#   hyphenat   [htt] lets TeX hyphenate inside \texttt. The report names a
#              lot of long identifiers (boltzmannOfP2Ppotential,
#              CONSISTENT_PARAMETER_CLOSURES) and without this they run past
#              the right margin -- 44 overfull boxes, the worst by 117pt.
# MiKTeX installs both on demand. The MSYS2 TeX Live in this tree does NOT
# have lmodern; install mingw-w64-x86_64-texlive-latex-extra there first.
set -e

for i in 1 2 3 4; do
    if [ "$i" = "2" ]; then
        bibtex ozGUI_ozLib_report > build_report_bib.log 2>&1 || true
    fi
    pdflatex -interaction=nonstopmode ozGUI_ozLib_report.tex \
        > "build_report$i.log" 2>&1 || true
done

if [ ! -f ozGUI_ozLib_report.pdf ]; then
    echo "BUILD FAILED - no PDF produced. First errors:"
    grep -A4 '^!' build_report4.log | head -20
    exit 1
fi

echo "built ozGUI_ozLib_report.pdf"
grep -E "Output written" build_report4.log || true
echo "errors:     $(grep -c '^!' build_report4.log || true)"
#MISSING FIGURES ARE ONLY A WARNING. pdflatex substitutes a draft box and
#carries on, so a report with every figure missing still reports zero
#errors and produces a plausible-looking PDF. That happened: the master had
#no \graphicspath while the figures lived in figures/, and nothing in this
#summary said so.
#Count missing FILES but not missing PACKAGES. A bare grep for "not found"
#also matches "File `lmodern.sty' not found", which is a LaTeX installation
#issue rather than a missing figure -- it reported MISSING FILES: 1 and sent
#someone hunting for a figure that was never missing. A check that cries
#wolf stops being read, which defeats the point of having it: this grep
#exists because a missing figure is only a WARNING in LaTeX, so the build
#reports success and produces a PDF with empty boxes.
missing=$(grep 'not found' build_report4.log | grep -vc '\.sty' || true)
if [ "$missing" != "0" ]; then
    echo "MISSING FILES: $missing  <-- figures will be EMPTY BOXES in the PDF"
    grep 'not found' build_report4.log | grep -v '\.sty' | head -5
fi
echo "undefined refs:      $(grep -ci 'undefined ref' build_report4.log || true)"
echo "undefined citations: $(grep -c 'Citation.*undefined' build_report4.log || true)"
echo "overfull boxes:      $(grep -c 'Overfull' build_report4.log || true)"

# A citation bibtex could not resolve shows as (?) in the PDF but does not
# fail the build, so surface bibtex's own complaints.
if grep -qi "error\|warning" build_report_bib.log 2>/dev/null; then
    echo "-- bibtex said:"
    grep -i "error\|warning" build_report_bib.log | head -5
fi
