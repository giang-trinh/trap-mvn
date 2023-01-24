`trap-mvn` is a Python package for computing various types of _trap spaces_ in multi-valued networks.

# Install

You will need the `clingo` ASP solver in your PATH. Instructions are provided directly on the [Potassco pages](https://github.com/potassco/clingo/releases/).

# Run `trap-mvn` from the command line

``` sh
$ python trapspace.py [-m maximum number of solutions] [-t maximum time to use in seconds] [-c type of trap spaces (min|max|fix)] [-s update semantics (general|unitary)] <SBML input file>
```

# Benchmark

All `.sbml` models used in our benchmark are given in the `sbml` folder.

You can re-run the benchmark via the Jupyter notebook `test.ipynb`.

The `AN-ASP` method includes two source files: `AN2asp.py` and `fixed-points.lp`.

The source code of Trappist is given in the `trappist` folder.

For the `mpbn` method, we need to install it using pip `pip install mpbn`.

We use `bioLQM.jar` to obtain the Booleanization of a multi-valued network following the Van Ham Boolean mapping.



