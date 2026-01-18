# MMF
This repository is the official implementation of "A Masked Mixture Model for More Expressive Matrix Factorization."


## Abstract
Matrix factorization (MF) is a widely used backbone for modeling large relational data due to its simplicity, scalability, and interpretability, yet its standard form relies on a single shared latent basis that can be too rigid for heterogeneous matrices.
In this paper, we propose Masked Mixture Factorization (MMF), which uses instance-specific masks that gate latent dimensions and form a mixture of masked components, enabling adaptive allocation of representational capacity while preserving MF’s efficient bilinear structure.
We further provide theoretical results on MMF’s expressivity and identifiability, clarifying when masking expands representational power and when the model is recoverable.
Under matched parameter budgets, MMF substantially increases effective expressivity and improves factorization accuracy over classical MF.
Extensive experiments on matrix reconstruction and matrix completion show consistent gains over strong baselines.

## Prerequisites
...
