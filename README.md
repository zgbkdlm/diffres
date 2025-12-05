# Diffusion differential resampling

This repository features implementation for the paper "Diffusion differential resampling".
In this paper, we have introduced a new differentiable-by-construction, consistent, and informative resampling method via diffusion models. 
Check out our preprint for details: https://xxx.

# Installation
Install via a standard procedure. 

```bash
git clone git@github.com:zgbkdlm/diffres.git
cd diffres

python -m venv .venv
python .venv/bin/activate
pip install -e .
```

# Example
Run any script in `./demos`.

# Reproduce experiments
The folder `./experiments` contain all the files that we use to generate the results. 
Each bash file is associated with on experiment. 
Check the `.sh` files to see how we run the experiments, and you can easily adapt them in your local machine.
Running them will *exactly* reproduce our results.

After running the experiments, you can use `./experiments/summary` which contains the plotting and printing scripts. 

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

# Contact
Zheng Zhao, Linköping University, https://zz.zabemon.com.
Jennifer R. Andersson, Uppsala University.
