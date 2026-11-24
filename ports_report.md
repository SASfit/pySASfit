| function | result | max rel. dev. | median | tolerance | C = 0 at | C [ms] | Python [ms] | speed-up | notes |
|---|---|---|---|---|---|---|---|---|---|
| ff_deformed_generalized_gaussian_chain | PASS | 3.9e-06 | 1.2e-15 | 1e-05 | 0 pts | 12.6 | 11.2 | 1.1 | exact everywhere; the C Guinier branch for sqrt(β)<0.1 deviates by up to ~3e-6 |
| ff_deformed_ring_polymer__bmb_ | FAIL | 2.4e+04 | 1.4e-08 | 1e-06 | 19 pts | 668.2 | 30.6 | 21.8 | tanh-sinh quadrature (~230 nodes) instead of adaptive sasfit_integrate |
| ff_deformed_ring_polymer__bz_ | FAIL | 8.2e+03 | 1.4e-08 | 1e-06 | 24 pts | 1460.1 | 35.7 | 40.9 | tanh-sinh quadrature (~230 nodes) instead of adaptive sasfit_integrate |
| ff_ringpolymerBMB | FAIL | 2.8e+03 | 1.4e-08 | 1e-06 | 11 pts | 590.0 | 28.5 | 20.7 | tanh-sinh quadrature |
| ff_ringpolymerBZ | FAIL | 8.3e+03 | 1.4e-08 | 1e-06 | 14 pts | 1345.6 | 39.0 | 34.5 | tanh-sinh quadrature |
