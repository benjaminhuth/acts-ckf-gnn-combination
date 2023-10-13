# GNN-CKF combined tracking

This is the repository related to https://indico.cern.ch/event/1252748/contributions/5521546.
Training scripts are not included, only final models in torchscript format.

NOTE: This is work in progress and subject to change.

## How to run

Driven by `snakemake`. To produce a desired output file, run
```
snakemake -c1 <filename>
```

## Requirements

* Custom ACTS branch: https://github.com/benjaminhuth/acts/tree/cdt2023
  * Compile with examples, python bindings, exatrkx plugin (torch backend), geant4, DD4hep, ODD.
  * Set environment variable `ODD_DIR` to the directory where the OpenDataDetector lives.
* Custom GNN4Itk common framework branch: [https://gitlab.cern.ch/gnn4itkteam/commonframework](https://gitlab.cern.ch/bhuth/commonframework/-/tags/cdt2023)
* ROOT (python interface) 6.28/02

## Issues?

Just contact me! Mail: benjamin.huth[/at/]ur.de

### Known issues
There can be situations in which the libtorch version and the python-managed pytorch version mismatch. 
In general it is not save to do `import torch` and `import acts` in the same python script.
If there are issues related to this, check `LD_LIBRARY_PATH`.
