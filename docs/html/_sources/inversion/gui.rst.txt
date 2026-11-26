Using the GUI
=============

Start
-----

.. code-block:: bash

   pysasfit-inversion [datafile | project.pysasfit]   # this tool alone
   pysasfit-gui [project.pysasfit]                    # all tools; "pyINVersion" tab
   python run_gui.py           # without installing

Projects and last session
-------------------------

**File → Save project…** writes a project file (``*.pysasfit``) with the
data (file name and an embedded copy of the numbers), the setting of every
control and the last result (background fit, inversion, bootstrap); **File →
Open project…** restores it, also when the data file has moved since (the
embedded copy is then written next to the project and loaded). A project
file given on the command line (``pysasfit-inversion proj.pysasfit``) is
opened at start.

The last session is saved the same way automatically (after loading data, a
background fit or a solve, and when the window closes) as
``session_inversion.pysasfit`` next to the preferences file and reopened at
the next start. **Settings → Forget last session** deletes it.

Projects carry a format version. Settings that a file does not contain --
for instance parameters added in a later version -- take their **default
values**, and entries the program does not know are ignored; the status bar
says which. See :mod:`pysasfit.core.project` for the file layout and the
compatibility rules.

Layout
------

The window consists of six **panels** (as in pySASred and pySASmorph): *Data*,
*Background* and *Inversion* (the controls), *Info*, *I(q)* and *Distribution*.
At the start the controls form the left column -- *Data* on top, *Background* and
*Inversion* as two tabs below it (Inversion in front) -- with the **Info** panel at
the bottom; on the right are :math:`I(q)` with error bars, background and fit
(log–log), with the normalised residuals :math:`(I - I_\mathrm{fit})/\sigma` of the fit
in a small panel below it (shared q axis, as in the SLD-profile tool), and the
recovered distribution with its uncertainty band and a dotted zero line.

Every panel can be moved by its title bar, stacked onto another panel as a tab or
put beside it, detached into its own window (first button in the title bar;
closing that window docks the panel again) and hidden (cross; *View* shows it
again). *View → Arrange panels*, *Dock all panels* and *Reset layout* sort them;
the title bar of the panel in use is coloured (Preferences → Appearance). The
arrangement is kept with the last session and in project files. Drag the border
above the info panel to change its height.

Info panel
    All explanations and all detailed results are shown in this single
    panel instead of in tooltips or long text blocks. It always shows the
    text of the control under the mouse pointer (the control or its label);
    with an open drop-down list it explains the highlighted entry (e.g. each
    solver or kernel). Moving away shows a short default text. Until its
    border is dragged, the panel is kept just large enough for the
    longest text at the current width, so the layout does not jump while
    the mouse moves; it never grows beyond that height, and if it is made
    smaller its text scrolls. Dragging it back to full height re-enables
    the automatic height. The
    static texts live in ``src/pysasfit/inversion/gui/help_texts.py`` (Qt rich text, one entry
    per control, solver, kernel and solver option).

    The control column itself only shows one-line summaries; the details
    appear in the info panel right after the action and again whenever the
    mouse is over the summary line:

    * **Data** (after loading): file and folder, number of points and
      skipped lines, :math:`q` range and spacing (points per decade for
      log-spaced data), :math:`I`, :math:`\Delta I` and
      :math:`\Delta I/I` ranges, :math:`D_{max}=\pi/q_{min}`, the
      resolution :math:`\pi/q_{max}`, the Shannon number and the number of
      data points per Shannon channel overall and in the upper third of the
      :math:`q` range (with a hint to use B-splines if it is below one).
    * **Background** (after a fit or with a fixed value): model, fit range
      and number of points, all parameters with errors, :math:`\chi^2_r` of
      the fit, the subtracted constant and its fraction of
      :math:`I(q_{max})`, and notes on extrapolation and degenerate
      parameters.
    * **Result** (after a solve): :math:`\chi^2_r` and its mean per third
      of the :math:`q` range, Pearson :math:`\chi^2/M`, Poisson deviance
      (G-test)/M and the KL divergence of the normalized curves, the
      estimated error scale :math:`s` (Bayesian evidence, option *Error
      bars: overall scale estimated*), solver and options, how the regularization
      strength was chosen (:math:`\lambda`, :math:`N_g`, :math:`\ln Z`,
      ...), the representation (grid, interpolation, B-splines, quadrature
      points), kernel and :math:`\alpha`, iterations, computing time,
      derived quantities (PDDF: :math:`I(0)=4\pi\int p\,dr`,
      :math:`R_g^2=\int p\,r^2dr/(2\int p\,dr)` and the position of the
      maximum; size distributions: maximum of :math:`x(r)`, number-weighted
      mean radius and maximum of the volume distribution), the type of
      error band, bootstrap statistics and warnings.

    The current suggestions for :math:`r_{max}` and the number of grid
    points are appended to the corresponding control texts.

Workflow
--------

1. **Load data file** -- three columns :math:`q`, :math:`I`,
   :math:`\Delta I`. :math:`r_{max}` and the number of grid points are
   suggested (:math:`D_{max}=\pi/q_{min}`, Shannon number), and are
   re-suggested when the kernel changes.
   **Error bars** -- from the file's :math:`\Delta I` column, or (always
   for files without one) an error model: estimated from local cubic fits
   through 5 points (default) or from neighbour interpolation, combined over
   a window of points, Poisson (:math:`\sigma^2=I`, counts only) or a relative error
   in percent (see :doc:`theory_forward_model`). The data info shows the
   ratio of the cubic-fit estimate to the file errors as a check.
2. **Background** -- fit :math:`I_{bg}+c\,q^{-\alpha}` in a :math:`q` range
   (only :math:`I_{bg}` is subtracted) or enter a fixed value.
3. **Kernel** -- the PDDF of a globular particle (:math:`4\pi j_0(qr)`),
   of the cross-section of a long cylinder (:math:`2\pi^2 J_0(qr)/q`) or of
   the thickness of a lamella (:math:`4\pi\cos(qr)/q^2`), all without
   :math:`\alpha`; or the sphere kernel for a radius distribution
   (:math:`\alpha=6` default). Choosing the lamella kernel switches the
   Bayesian-evidence penalty to "p(0) free" (and back to eq. 19 for the
   other kernels). The result panel shows the forward value and the radius
   of gyration of the particle, cross-section or thickness.
4. **Grid** -- :math:`r_{max}`, number of points, spacing (uniform,
   quadratic, logarithmic) and interpolation between the points (linear or
   cubic). Spacing and interpolation are hidden in B-spline mode;
   interpolation is fixed to linear for the solvers that need
   :math:`x\ge0`; the list is then greyed out and marked
   *(linear only for this solver)*.
5. **Solver** and its options (appear only for the solver that has them),
   optionally "Force :math:`\chi^2_r=1`" (Bayesian evidence).
6. **B-splines** -- "Represent p(r) by B-splines (any solver)", number of
   coefficients and knot spacing.
7. **Monte Carlo uncertainty** -- parametric bootstrap with :math:`N`
   replicates.
8. **Solve**.

Result line
-----------

The one-line summary below *Solve* shows :math:`\chi^2_r`, the number of
bootstrap replicates if used, and a warning sign if something needs
attention; everything else is in the info panel (see above). Warnings
include a solver that needs a non-negative kernel used with a signed
kernel, and a :math:`\chi^2_r=1` target that was not reachable.

Plots
-----

The distribution plot shows the MAP solution with the analytic
:math:`\pm1\sigma` band for the Bayesian-evidence solver, or the bootstrap
mean with its :math:`\pm1\sigma` band and the single solve (dashed) when
the Monte Carlo option is used. The :math:`y` axis is :math:`p(r)` for the
PDDF kernel and :math:`x(r)=N(r)\,r^{\alpha}` for size distributions. The
zero line style is set by ``ZERO_LINE_STYLE`` in ``app.py``.
