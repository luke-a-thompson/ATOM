Chemical Graph Transformer Neural Operator (CGTNO) is a neural operator for learning point cloud dynamics with specific application to molecular dynamics.

The notation in the paper generally corresponds to our comments, with the following caveats:
* Timesteps - P -> T

Please install with `poetry install --with dev` if you want type checking (i.e., for mypy).

# Data sources
The same as used in EGNO: https://www.sgdml.org/

Recommended further reading:
1. EGNO: https://arxiv.org/pdf/2401.11037
    * General graph neural operator architecture
1. https://arxiv.org/pdf/2203.06442
    * Dataset setup in more detail
1. http://www.sgdml.org/
    * Dataset source
1. https://arxiv.org/pdf/2302.14376
    * Heterogenous attention - Gave inspiration for our approach
1. https://arxiv.org/pdf/2108.08481
    * Neural operator theory