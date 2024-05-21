# cell2module

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/vitkl/cell2module/test.yaml?branch=main
[link-tests]: https://github.com/vitkl/cell2module/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/cell2module

cell2module: gene module discovery from scRNA and scATAC using count-based Bayesian NMF

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install cell2module:

<!--
1) Install the latest release of `cell2module` from `PyPI <https://pypi.org/project/cell2module/>`_:

```bash
export PYTHONNOUSERSITE="aaaaa"
conda create -y -n cell2module_v01 python=3.9
conda activate cell2module_v01
pip install cell2module
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/vitkl/cell2module.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/vitkl/cell2module/issues
[changelog]: https://cell2module.readthedocs.io/latest/changelog.html
[link-docs]: https://cell2module.readthedocs.io
[link-api]: https://cell2module.readthedocs.io/latest/api.html
