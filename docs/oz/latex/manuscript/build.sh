#!/bin/sh
# Build the manuscript PDF.
#
#   sh build.sh
#
# FOUR passes with bibtex between them, not two. The sequence matters:
#   pdflatex   writes the .aux, recording which entries are cited
#   bibtex     reads the .aux and ../ozgui.bib, writes the .bbl
#   pdflatex   pulls the .bbl in
#   pdflatex   resolves the citation labels and the table/figure refs
# Skip the bibtex pass and every citation renders as (?) with an otherwise
# clean build, which is easy not to notice.
#
# The bibliography is ../ozgui.bib, shared with the report. Add entries
# there, not here.
#
# NOTE microtype uses [expansion=false]: font expansion needs scalable fonts,
# and without the option a minimal TeX installation fails with
#   ! pdfTeX error: auto expansion is only possible with scalable fonts.
# producing NO PDF at all. The report needs lmodern for a related reason --
# see docs/build_report.sh.
set -e

cp -f figures/*.pdf figures/*.png . 2>/dev/null || true

pdflatex -interaction=nonstopmode manuscript.tex > build1.log 2>&1 || true
bibtex manuscript                                > build_bib.log 2>&1 || true
pdflatex -interaction=nonstopmode manuscript.tex > build2.log 2>&1 || true
pdflatex -interaction=nonstopmode manuscript.tex > build3.log 2>&1 || true

if [ ! -f manuscript.pdf ]; then
    echo "BUILD FAILED - no PDF produced. First errors:"
    grep -A4 '^!' build3.log | head -20
    exit 1
fi

echo "built manuscript.pdf"
grep -E "Output written" build3.log || true
echo "errors: $(grep -c '^!' build3.log || true)   undefined refs: $(grep -ci 'undefined ref' build3.log || true)   overfull: $(grep -c Overfull build3.log || true)"

# A citation that bibtex could not resolve shows as (?) in the PDF but does
# not fail the build, so report it explicitly.
undefcite=$(grep -c "Citation.*undefined" build3.log || true)
echo "undefined citations: $undefcite"
if [ "$undefcite" != "0" ]; then
    echo "  -> check ../ozgui.bib for the missing keys, and that bibtex ran:"
    grep -i "error\|warning" build_bib.log | head -5
fi
