# `trapmvn`: Trap space detection in multi-valued networks

Package `trapmvn` implements trap space detection in multi-valued 
logical models through answer set programming. Currently, we support 
[SBML-qual](https://sbml.org) (used e.g. by [GINsim](http://ginsim.org)) 
and [BMA](http://biomodelanalyzer.org) model formats as inputs.

You can install `trapmvn` through `pip`:

```
pip install git+https://github.com/giang-trinh/trap-mvn.git
```

### Command line usage

You can use `trapmvn` from command line as a standalone program. The
output should be a valid `.tsv` file (tab-separated-values) representing
all trap space. The program accepts the following arguments:

 - `-c` [`--computation`]: Use `min`, `max` or `fix` to compute minimal trap
 spaces, maximal trap spaces, or fixed-points.
 - `-s` [`--semantics`]: Use `unitary` or `general` to define the desired 
 model variable update scheme.
 - `-m` [`--max`]: Integer limit for the number of enumerated solutions.
 - `-fm` [`--fixmethod`]: Use either `deadlock` and `siphon` to switch between
 different fixed-point computation methods.

The input model can be either given on standard input (in which case the presumed
format is SBML), or as a file path in the last argument (in which case we infer
the format from the file extension).

Example usage:

```
python3 -m trapmvn --computation max --semantics general --max 10 ./path/to/model.bma
python3 -m trapmvn -c min -s unitary -m 100 ./path/to/model.sbml
```

### Python API and case study

If you want to use `trapmvn` directly from Python, you can inspect Jupyter notebooks
in our `case-study` folder. Here, we show how to load a model, convert it into a
Petri net encoding and subsequently compute trap spaces using the `trapmvn` method.

The case study itself is concerned with assessing the reliability of knockout 
interventions in a large-scale model of Myc-deregulation in breast cancer.

### Benchmarks

TODO
 