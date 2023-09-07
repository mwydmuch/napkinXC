# napkinXC experiments

This directory contains scripts to replicate experiments from following papers:
- `plt_jmlr` - [Probabilistic Label Trees for Extreme Multi-label Classification](https://arxiv.org/abs/2009.11218)
- `oplt_aistats` - [Online probabilistic label trees](http://proceedings.mlr.press/v130/jasinska-kobus21a.html)
- `psplt_sigir` - [Propensity-scored Probabilistic Label Trees](https://dl.acm.org/doi/10.1145/3404835.3463084)

It uses a combination of Bash and Python scripts. Each file contains one kind of experiment.
Experiments that are defined in shell files (.sh) assume that the napkinXC executable (nxc) is present in the repository root directory.
Python scripts assume that napkinxc package can be imported.

