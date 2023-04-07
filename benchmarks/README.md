## Benchmarks

 > The benchmark scripts were only tested on Linux (even though `trapmvn` itself is multi-platform).

 > We include the `bioLQM.jar`, `an-asp` and `trappist` directly in this folder, but you still need to install `mpbn` manually (`pip install mpbn`). You also need to have `clingo` installed as a standalone program.

This folder contains Python scripts related to performance evaluation of `trapmvn`:
 - `bench_fix.py`, `bench_min.py`, `bench_max.py` contain code necessary to perform a specific experiment.
 - Where applicable, the scripts measure both "time to first result" and "time to all results" measurements.
 - `bma_to_sbml.ipynb`notebook performs conversion of BMA models to SBML in order to ensure comparable inputs for all tools. The converted models are already included in the repository, but you can use this notebook to recreate them.

Finally, `old_benchamrks.ipynb` contains an older version of the benchmarks, performed on i9-11950H CPU using a slightly older version of `trapmvn`. The latest results are different due to 
the different hardware used, but the general trend is the same in both datasets.