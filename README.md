# EclipseNETs: Learning Irregular Small Celestial Body Silhouettes
![4](https://github.com/user-attachments/assets/5c8f3a8b-8a3e-47ad-8033-7ad31ed968db)
This repository contains the official implementation of the paper:

> [**EclipseNETs: Learning irregular small celestial body silhouettes**](https://www.sciencedirect.com/science/article/abs/pii/S0094576525003443), 
> *Acta Astronautica*, 2025.

## Project Overview

Traditional eclipse modeling for spacecraft trajectory design around asteroids and other irregular small celestial bodies relies on expensive ray-tracing operations against high-fidelity shape models (e.g., polyhedral meshes). These methods are computationally prohibitive for real-time applications and large-scale trajectory optimization.

**EclipseNETs** address this challenge by training neural models to predict differentiable irregular **silhouettes** (i.e., the 2D projected shadow of the body as seen from the spacecraft) as a function of spacecraft position and orientation relative to the body's center of mass. The key innovations include:

- **Fully differentiable silhouette prediction**: Enables end-to-end gradient-based optimization of trajectories that account for eclipses.  
- **Fast inference**: Learned models evaluate in microseconds, enabling also much faster integrations of orbital trajectories with eclipse events.  
- **High fidelity**: Achieves high accuracy in reproducing the actual eclipse.
- **Integration with Taylor-based integrator**: they can be used as event detection functions inside Taylor propagators, enabling differentiable and reliable event detection.

The approach unlocks **differentiable eclipse description** for proximity operations around small bodies, where accurate shadowing prediction is critical but computationally challenging.
![7](https://github.com/user-attachments/assets/1516c65e-e3bf-4229-b788-98d509a70c92)



## Installation

```
conda env create -f environment.yml
conda activate eclipsenets
pip install -e .
git lfs pull   # downloads the datasets (about 1.3 GB)
```

The shape models, the mascon models and the trained networks are part of the repository. The datasets are stored with [git lfs](https://git-lfs.com): they are needed to train the networks and to reproduce Table 2 and Figure 3, not to use the trained networks.

## Quick start

```python
import numpy as np
import eclipsenets as en

body = en.load_body("67P")                 # shape model, mascon model, physical constants
model = en.load_pretrained("67P", "siren")  # the EclipseNET of the paper

# Orbit propagation, with the network detecting the eclipses
propagator = en.EclipseNetPropagator(body, model)
ic = en.initial_state(radius=1.5, inclination=np.radians(11.), omega=body.omega)
propagation = propagator.propagate(ic, np.linspace(0, 3 * np.pi, 240), sun_direction=[-1., -1., 0.])
print(propagation.eclipses)                # start and end time of each eclipse
```

## Notebooks

The notebooks in `notebooks/` reproduce the results of the paper:

| Notebook | Content | Paper |
|---|---|---|
| `1_dataset_generation` | silhouettes, eclipse function and its samples, datasets | Figure 2 (A-C) |
| `2_training_and_evaluation` | the four bodies, losses of the networks, predicted silhouettes, training | Tables 1 and 2, Figures 2 (D) and 3 |
| `3_orbit_propagation` | the network as event function of a Taylor integrator, against ray tracing | Figure 4 |
| `4_neuralode_refinement` | refinement of a network from trajectory data | Figure 5 |

## Differences from the paper

The notebooks reproduce the results of the paper, with these differences, which do not change techniques/conclusions/main message of the paper, but are worth noting:

- **Size of the datasets.** The paper reports 22 million training points and 9 million validation points per body. The datasets in `datasets/` are about ten times smaller: 2.3 million and 0.9 million points (500 and 200 Sun directions, with about 4,500 points each).
- **Table 2.** The losses of Bennu, 67P and Eros are identical to the ones of the paper. The datasets of Itokawa have been generated again, and the losses of its networks are within 4% of the ones of the paper.
- **Figure 3.** The networks of 67P are trained again, from a different random initialization. The SIREN ends with a lower loss than in the paper (2.9e-5 against 4.6e-5), the feed-forward network with a similar one (8.4e-5 against 8.0e-5).
- **Figure 5.** The error of the trajectory before the refinement is about 3 m, against about 58 m in the paper, and the refinement reduces it by a factor of 18 rather than 7.

## Repository

| Path | Content |
|---|---|
| `eclipsenets/` | the library |
| `notebooks/` | the notebooks above |
| `scripts/` | `generate_dataset.py` and `train.py`, to build a dataset and train a network from the command line |
| `tests/` | unit tests (`pytest`) |
| `3dmeshes/`, `mascons/` | shape models and mascon models of Bennu, Itokawa, 67P and Eros |
| `datasets/` | training and validation sets (git lfs) |
| `models/` | the trained networks |

The library is organized as follows:

| Module | Content |
|---|---|
| `bodies` | `load_body`, `load_mesh`, `load_mascons` |
| `silhouette`, `eclipse_function` | `silhouette_polygon`, `compute_eclipse_function` |
| `sampling`, `dataset` | `fibonacci_sphere`, `sample_view`, `generate_dataset`, `load_dataset` |
| `nn`, `training` | `SIREN`, `FFNN`, `load_pretrained`, `load_model`, `heyoka_model`, `train`, `evaluate_mse` |
| `ray_tracing` | Möller-Trumbore intersection tests, `is_eclipsed` |
| `dynamics` | `EclipseNetPropagator`, `propagate_ray_tracing`, `is_eclipsed_net` |
| `neural_ode` | `state_sensitivities`, `refine` |
| `utils`, `plots` | projections on the plane orthogonal to the Sun direction, plotting helpers |


## Citation

If you use this work, please cite the following publication:

```bibtex
@article{acciarini2025eclipsenets,
title = {EclipseNETs: Learning irregular small celestial body silhouettes},
journal = {Acta Astronautica},
volume = {236},
pages = {514-521},
year = {2025},
issn = {0094-5765},
doi = {https://doi.org/10.1016/j.actaastro.2025.06.002},
url = {https://www.sciencedirect.com/science/article/pii/S0094576525003443},
author = {Giacomo Acciarini and Dario Izzo and Francesco Biscani},
keywords = {Small bodies, Silhouette reconstruction, Asteroids, Comets, NeuralODE, Neural events, Machine learning, Artificial intelligence, AI for space, Spaceflight mechanics, Orbital dynamics},
}
```
