pySASfit logo bundle
====================

Tools (working names as agreed in the design chat)
  pySASfit      suite master logo: fan of fit curves in the colours of the other tools
  pySASred      data reduction (detector rings to I(q))
  pyOZsolver    Ornstein-Zernike structure factors
  pyINVersion   Fredholm inversion, size distributions   (wordmark: py + INV + ersion)
  pySLDprofile  SLD distributions
  pySASstudio   model fitting, single and later multiple data sets
  pySCAtter     scattering patterns of particles and lattices (glyph: diffraction spots; tile sienna #A0522D)
  pySEBpolymers structures of linked sub-units (glyph: core with grafted chains; tile olive #6B8E23)
  pySASmorph    bicontinuous structures from clipped random waves (two phases + interface layer; multi-phase planned)
  pyTOOLbox     small helpful tools

Folders
  icons/<tool>/       square tile for software (window/app/favicon)
      <tool>-icon.svg           master vector
      <tool>-icon-small.svg     thicker strokes, for 16-32 px use
      png/                      16, 24, 32, 48, 64, 128, 256, 512, 1024 px, transparent corners
      <tool>.ico                Windows icon (16-256 px, PNG-encoded)
  logos/<tool>/       horizontal logo = tile + wordmark, for documentation, web, slides
      svg/                      light, dark, mono-black, mono-white  (text converted to outlines)
      png/                      same four variants, 256 px and 1024 px high, transparent background
  print/<tool>/       vector PDFs (colour on white, one-colour black) for printed material

Which variant where
  light        white or light backgrounds (documentation, papers, posters)
  dark         dark backgrounds (uses a lighter blue tile and white text)
  mono-black   one-colour print, stamps, fax-style or greyscale documents
  mono-white   one-colour on dark or photo backgrounds
  Icons below 32 px: use the -small version, it has heavier strokes.

Colours
  Navy tile            #0E2740
  Navy tile on dark    #153A5E
  Wordmark core        #0E2740 (white on dark)
  "py" grey            #5E7388 (#9FB6CC on dark)
  Word ending (suffix) same as "py": #5E7388 on light, #9FB6CC on dark
  Data cyan            #7FD3E8
  pySASfit tile        sky blue #CFE8FB
  pySASfit curve fan   (= tile colours of the other tools, navy lead curve)  #6B46C1 #2E8B57 #C23A93 #2F6FED #D9730D #C8283C #0E2740
  Typeface             Space Grotesk (SIL Open Font License), already outlined in all logo files

Use
  Keep clear space of at least a quarter of the tile height around a logo.
  Do not stretch or recolour the tile glyphs; use the mono versions instead.
  Minimum sizes: icon 16 px, logo 24 px high on screen, 8 mm high in print.

Print note
  The PDFs are vector and in RGB. For professional offset printing, let the printer or
  Inkscape/Scribus convert to the target CMYK profile, or tell me the profile and spot
  colours you need.

Names and colours (logos/)
  The logos use the same coloured tile as the menu icons. In the wordmark, "py" is grey, the core
  of the name takes the tile colour (pyOZsolver: OZ in violet), and the word ending has the same
  grey as "py" for all tools (#5E7388 on light, #9FB6CC on dark backgrounds).
  On dark backgrounds the core colour is lightened by about a third for contrast.
  The navy icons in icons/ (all tools except pySASfit, which uses the sky-blue tile there too)
  are the earlier detailed app-icon design, kept as an alternative.
  In the pySASfit name the core "SAS" is navy, because the sky-blue tile is too light for text.

Menu and toolbar icons (menu-icons/)
  For small sizes (menus, toolbars, taskbars, file lists) use menu-icons/<tool>/, not icons/.
  Each tool has its own tile colour and one bold shape, so they can be told apart at 16 px:
    pySASfit      sky blue #CFE8FB  converging curves, each in the colour of another tool
    pySASred      red #C8283C    concentric detector rings
    pyOZsolver    violet #6B46C1 g(r): contact spike with damped oscillation
    pyINVersion   green #2E8B57  size-distribution histogram
    pySLDprofile  magenta #C23A93 stepped SLD profile
    pySASstudio   blue #2F6FED   data points with fit curve
    pySCAtter     sienna #A0522D hexagonal diffraction spots around a gold centre
    pySEBpolymers olive #6B8E23  gold core with six grafted wavy chains
    pySASmorph    marine #0E2740 two bicontinuous phases (marine, amber) + white interface layer
    pyTOOLbox     orange #D9730D open box with small tools
  Files per tool: <tool>-menu.svg, png/ (16, 20, 24, 32, 40, 48, 64, 128, 256 px), <tool>-menu.ico,
  and glyph/ with transparent one-colour glyphs (black and white, SVG and 16/24/32 px PNG) for
  menu bars and toolbars that recolour icons themselves.
  The larger icons/ and logos/ keep the navy family tile; the menu icons are the small-size companion.
