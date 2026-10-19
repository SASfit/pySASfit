# Data sets for testing

This folder is for storing datasets to run the analysis as we start the multifit project with FitBenchmarking.

## [SANS_polymer_single](https://github.com/fitbenchmarking/sasfit-collab/tree/main/data/SANS_polymer_single)

- ASCII file [D0025674.021](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_polymer_single/D0025674.021) with header
- columns: q, I(q), Delta I(q), Delta Q, i.e. x,y, error, resolution
- SASfit project file: SANS_polymer_single/SANS_polymer_single.sas
- resolution is not determined and 4th column is therfore 0
- model: 
  1. background: c0=0.057
  2. generalized Gaussian coil: Rg=4.11, nu=0.523, I0=1.255
- SASfit project file modelling the data [SANS_polymer_single.sas](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_polymer_single/SANS_polymer_single.sas)

<a href="SANS_polymer_single.sas"><img src="https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_polymer_single/SANS_polymer.png" width="500" ></a>

## [SANS_SwollenMicrogel](https://github.com/fitbenchmarking/sasfit-collab/tree/main/data/SANS_SwollenMicrogel)

- ASCII file [SANS_SwollMicrogel.dat](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_SwollenMicrogel/SANS_SwollMicrogel.dat) without header

- columns: q, I(q), Delta I(q), Delta Q, i.e. x,y, error, resolution

- resolution is not determined and 4th column is therfore 0

- two models have been applied to model the data. 
  
  1. First model is a polydisperse fuzzy sphere model with internal inhomogeneieties for which a blob scattering is added together with an incoherent constant background. The corresponding SASfit project file modelling the data: [SANS_SwollMicrogel_FuzzySpherePlusBlob.sas](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_SwollenMicrogel/SANS_SwollMicrogel_FuzzySpherePlusBlob.sas)
  
  <a href="fuzzy_sphere_plus_blob_scattering.png"><img src="https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_SwollenMicrogel/fuzzy_sphere_plus_blob_scattering.png" width="500" ></a>
  
  2. A second model uses the matalog distribution to generate a flexible monotonic decaying profile which is consistent with the data. No polydispersity is assumed in this case. However, the model seems to have a lot of local minima due to the flexibility of the metalog distribution. Also here inhomogeneieties are described by blob scattering together with a small constant background.
     The corresponding SASfit project file modelling the data:  [SANS_SwollMicrogel_MetalogProfilePlusBlob.sas](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_SwollenMicrogel/SANS_SwollMicrogel_MetalogProfilePlusBlob.sas)
  
  <a href="metalog_sphere_plus_blob_scattering.png    "><img src="https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_SwollenMicrogel/metalog_sphere_plus_blob_scattering.png" width="500" ></a>

## [SASfitTestData](https://github.com/fitbenchmarking/sasfit-collab/tree/main/data/SASfitTestData)

- ASCII file [test.dat](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SASfitTestData/test.dat) without header
- three columns: q, I(q), Delta I(q), i.e. x,y, error
- a model describing the data is a bimodal size distribution of spheres with a constant incoherent background. 
  1. first population are larger polydisperse spheres with a log normal radius distribution
  2. second polpulation are smaller polydisperse spheres with a log normal distribution
  3. a constant q-independent background scattering
- SASfit project file modelling test data set:  [SASfit_test.sas](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SASfitTestData/SASfit_test.sas) 

<a href="SASfit_test.png"><img src="https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SASfitTestData/test_dat.png" width="500" ></a>

## [SANS_cv_STJ1](https://github.com/fitbenchmarking/sasfit-collab/tree/main/data/SANS_cv_STJ1)

- ASCII files without header
- four columns: q, I(q), Delta I(q), Delta q, i.e. x,y,error, resolution
- contrast variation series with 9 contrasts. 
- all parameters except the scattering length density of solvent are global parameters.
- the data have been published in [J. Chem. Phys. 125, 044715 (2006)](https://doi.org/10.1063/1.2220564)
- SASfit project file modelling the data [STJ1_5%.sas](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_cv_STJ1/STJ1_5%.sas)

<a href="STJ1_5%.png"><img src="https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SANS_cv_STJ1/STJ1.png" width="500" ></a>

## [SESANS_polystyrene2micron](https://github.com/fitbenchmarking/sasfit-collab/tree/main/data/SESANS_polystyrene2micron)

- ASCII file in ses-format used for SESANS [se008731_01.ses](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SESANS_polystyrene2micron/se008731_01.ses)
- Polystyrene 2 um in 53% H2O, 47% D2O 
- SASfit project file modelling the data [SESANS_polystyrene2micron.sas](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SESANS_polystyrene2micron/SESANS_polystyrene2micron.sas)

<a href="SESANS_polystyrene2micron.sas"><img src="https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SESANS_polystyrene2micron/SESANS_polystyrene2micron.png" width="500" ></a>

## [SESANS_core_shell](https://github.com/fitbenchmarking/sasfit-collab/tree/main/data/SESANS_core_shell)

- ASCII file in ses-format used for SESANS [se009051.ses](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SESANS_core_shell/se009051.ses)
- SASfit project file modelling the data [se009051_core_shell.sas](https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SESANS_core_shell/se009051_core_shell.sas) 

<a href="SASfit_test.sas"><img src="https://github.com/fitbenchmarking/sasfit-collab/blob/main/data/SESANS_core_shell/SESANS_core_shell.png" width="500" ></a>