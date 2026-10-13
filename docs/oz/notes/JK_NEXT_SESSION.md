- work on a new NEXT_SESSION.md and update its contents if task are already successfully done, tested, and documented

- also consider to add:
  
  - The manual says "Anderson acceleration (SUNDIALS KINSOL KIN_FP) is tried first, with Newton-Krylov (GMRES with line search) and damped Picard iteration as successive fallbacks." Shouldn't we do the same in section53_survey. 
  
  - We also should note the speed, each strategy needs to solve the OZ equations as this could help the user to decide for which task which strategy give first relevant info.
  
  - For size distributions of large width especially the heavy tailed distributions like lognormal need care, that the size distribution is done over a significant large interval. I would assume an integration routine specialized on intervals from 0 to infinity might be here the choice, where they might fail for extremely narrow size distributions. In thoses cases with very narroe distributions, it might be better to integrate over mean+-n*variance. 
  
  - section53_survey.py finished on my local machine and results are ready for download

- update manuscript.

- update report. The report are two latex documents. They should joined to one like the manuscript, i.e. a master is including two tex files.

For those closures optimizing thermodynamic consistency the algorithm looks for a sign change. If none can be found the fallback is to find the smalles residual. However, the correct fallback would be to find the smallest modulus of the residual, i.e. min{fabs(residual)}. It claimes using closest-residual-to-zero but it rather  uses smallest residual. 

I still like to have the option to solve via iteration and fix point strategy rather than classes, mainly for education purposes.
