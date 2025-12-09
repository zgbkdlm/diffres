# Diffusion differential resampling

This repository features implementation for the paper "Diffusion differential resampling".
In this paper, we have introduced a new **differentiable-by-construction**, **consistent**, and **informative** resampling method via diffusion models. 
Check out our preprint for details at https://arxiv.org/abs/xxxx.xxxxx.

# Installation
Install the package via a standard procedure. 

```bash
git clone git@github.com:zgbkdlm/diffres.git
cd diffres

python -m venv .venv
python .venv/bin/activate
pip install -e .
```

# Example
We have provided a few examples for demonstration in folder `./demos`, e.g., the Jupyter Notebook `gaussian_mixture.ipynb`.

# Quick start
Suppose that you have weighted samples and wish to do differentiable resampling, the simplest code using the diffusion resampling would look like this:

```python
import jax
from diffres.resampling import diffusion_resampling

key = jax.random.PRNGKey(666)

# The given weighted samples
samples = ...
log_ws = ...

# Resampling parameters
a = ...
ts = ...

# Resampling
_, resamples = diffusion_resampling(key, log_ws, samples, a, ts)
```

# Reproduce experiments
The folder `./experiments` contain all the files that we use to generate the results. 
Each bash file is associated with one experiment. 
Check the `.sh` files to see how we ran the experiments, and you can easily adapt them in your local machine.
Running them will *exactly* reproduce our results in the paper.

After running the experiments, you can then run the scripts in `./experiments/summary` to print and plot the results. 
These scripts also guarantee to *exactly* reproduce the tables and figures in the paper. 

# Citation
Please cite using the following bibtex. 

```bibtex
@article{Andersson2025diffres, 
    author = {Andersson, Jennifer R. and Zhao, Zheng}, 
    title = {Diffusion differentiable resampling},
    journal = {arXiv preprint arXiv:xxxx.xxxxx},
    year = {2025},
}
```

# License
GPL v3 or later. 
For those aiming proprietarisation, please invest your own time to re-implement the algorithm by yourself. 

# Contact
Zheng Zhao, Linköping University, https://zz.zabemon.com.
Jennifer R. Andersson, Uppsala University.
