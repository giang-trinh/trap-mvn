## Benchmarks

 > The benchmark notebooks were only tested on Linux (even though `trapmvn` itself is multi-platform).

 > We include the `bioLQM.jar`, `an-asp` and `trappist` directly in this folder, but you still need to install `mpbn` manually (`pip install mpbn`). You also need to have `clingo` installed as a standalone program.

This folder contains Jupyter notebooks related to performance evaluation of `trapmvn`:
 - `bma_to_sbml.ipynb`: Conversion of BMA models to SBML in order to ensure comparable inputs for all tools. The converted results models
 are already included in the repository, but you can use this notebook to recreate them.
 - `fixed_points.ipynb`: Compares `trapmvn` with `mpbn`, `trappist` and `an-asp` on the problem of
 fixed-point computation. Furthermore, it compares the performance of `deadlock` and `siphon` problem formulations. 
 The results are saved in `fixed-point-benchmark.tsv`.
 - `minimal_trap_spaces.ipynb`: Compares `trapmvn` with `mpbn` and `trappist` on the problem
 of minimal trap space computation. Also includes `trapmvn` with general semantics (as opposed to unitary).
 The results are saved in `min-trap-benchmark.tsv`.
 - `maximal_trap_space.ipynb`: Benchmark of maximal trap space computation using `trapmvn` on general
 and unitary semantics. The results are saved in `max-trap-benchmark.tsv`.
 

The times presented in all notebooks were obtained running exclusively on one core of a i7-4790 CPU with 32GB of RAM (vast majority of experiments should not come anywhere close to this limit though).

Finally, `old_benchamrks.ipynb` contains an older version of the benchmarks, performed on i9-11950H CPU using a slightly older version of `trapmvn`. The latest results are different due to 
the different hardware used, but the general trend is the same in both datasets.