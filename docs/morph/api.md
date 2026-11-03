# pySASmorph: API reference

Generated from the docstrings.

```{eval-rst}
.. automodule:: pysasfit.morph.pipeline
   :members: MorphParameters, MorphResult, reconstruct, correlations, suggest_box, porosity_from_invariant

.. automodule:: pysasfit.morph.multiphase
   :members: DataSet, PhaseSystem, MultiCut, MultiParameters, CurveResult, MultiResult, correlations_multi,
             reconstruct_multi, fit_fractions, joint_g, interface_areas, multiphase_statistics

.. automodule:: pysasfit.morph.extrapolation
   :members: ExtrapolatedCurve, extrapolate, fit_porod, fit_low_q, auto_porod_range

.. automodule:: pysasfit.morph.correlation
   :members: LevelCut, spectral_function, wavevector_distribution

.. automodule:: pysasfit.morph.grf
   :members: field_fft, field_cosine, level_cut, cut_level, grid_distances, power_spectrum_on_grid

.. automodule:: pysasfit.morph.transforms
   :members: sine_transform_pl, moments_pl, porod_tail

.. automodule:: pysasfit.morph.scattering
   :members: voxel_intensity, levelcut_intensity, contrast_from_invariant

.. automodule:: pysasfit.morph.statistics
   :members: specific_surface_porod, specific_surface_voxels, phase_connectivity, morphology_statistics

.. automodule:: pysasfit.morph.export
   :members: write_curves, write_matrix, write_vtk, write_hdf5, write_tiff_stack, write_npy, write_curves_multi,
             write_hdf5_multi, write_labels

.. automodule:: pysasfit.morph.surface
   :members: Mesh, phase_surface, label_surface, largest_cluster, vertices_in, write_stl, marching_cubes_backend

.. automodule:: pysasfit.morph.view3d_gl
   :members: MorphGLView, gl_problem, headlight_shader

.. automodule:: pysasfit.morph.view3d
   :members: pyvista_problem, have_pyvista, surface_mesh, show_3d

.. automodule:: pysasfit.morph.anisotropic
   :members: Data2D, load_2d, AnisoParameters, AnisoResult, estimate_axis, axis_coordinates, legendre_components,
             spherical_transform, fit_affine, correlations_aniso, reconstruct_aniso, field_from_grid_correlation,
             chord_lengths, folded_map

.. automodule:: pysasfit.morph.gui
   :members: MainWindow, main

.. automodule:: pysasfit.morph.aniso_gui
   :members: AnisoWindow, main
```
