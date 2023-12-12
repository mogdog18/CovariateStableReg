# Ensuring Superiority of Stable Regression in the Presence of Covariate Shift
### Authors: Anna Midgley, Nora Hallqvist
This is the final project for 15.095 Machine Learning Under a Modern Optimization Lens.

## Abstract
The full report for the project can be found under `src/report.pdf`. The abstract is as follows:
We investigate the performance of Stable Regression, compared to the random assignment of data to training and validation 
sets in the presence of a covariate shift.  In particular, our investigation reveals that without the integration of 
importance sampling, Stable Regression is not guaranteed to outperform randomization under a covariate shift. 
However, when importance sampling is applied to both methods, Stable Regression regains it superiority, yielding more accurate and stable results. 

## Code Base
The code base for this project is located under `src/`. The solvers are defined in
`src/regressionClass.jl`, and are called in `src/get_results.ipynb`. Various utility 
functions are defined in `src/utils.jl`. The data used in the experiments is located
in `data`, and is either synthetically produced or taking from the UCI Machine Learning Repository.