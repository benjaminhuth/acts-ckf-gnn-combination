# GNN-CKF combined tracking

This is the repository related to https://indico.cern.ch/event/1252748/contributions/5521546.
Training scripts are not included, only final models in torchscript format.
 https://gitlab.cern.ch/bhuth/commonframework/-/tags/cdt2023

NOTE: This is work in progress and subject to change.

## How to run

Driven by `snakemake`. To produce a desired output file, run
```
snakemake -c1 <filename>
```

## Requirements

* Custom ACTS branch: https://github.com/benjaminhuth/acts/tree/cdt2023
  * Compile with examples, exatrkx plugin (torch backend), geant4, DD4hep, ODD.
* Custom GNN4Itk common framework branch: [https://gitlab.cern.ch/gnn4itkteam/commonframework](https://gitlab.cern.ch/bhuth/commonframework/-/tags/cdt2023)

## Issues?

Just contact me! Mail: benjamin.huth[/at/]ur.de
