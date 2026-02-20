# EclipseNETs: Learning Irregular Small Celestial Body Silhouettes
![4](https://github.com/user-attachments/assets/5c8f3a8b-8a3e-47ad-8033-7ad31ed968db)
This repository contains the official implementation of the paper:

> [**EclipseNETs: Learning irregular small celestial body silhouettes**](https://www.sciencedirect.com/science/article/abs/pii/S0094576525003443), 
> *Acta Astronautica*, 2025.

## Project Overview

Traditional eclipse modeling for spacecraft trajectory design around asteroids and other irregular small celestial bodies relies on expensive ray-tracing operations against high-fidelity shape models (e.g., polyhedral meshes or spherical harmonics). These methods are computationally prohibitive for real-time applications and large-scale trajectory optimization.

**EclipseNETs** address this challenge by training *differentiable neural networks* to predict irregular **silhouettes** (i.e., the 2D projected shadow of the body as seen from the spacecraft) as a function of spacecraft position and orientation relative to the body's center of mass. The key innovations include:

- **Fully differentiable silhouette prediction**: Enables end-to-end gradient-based optimization of trajectories that account for eclipses.  
- **Fast inference**: Learned models evaluate in microseconds, enabling also much faster integrations of orbital trajectories with eclipse events.  
- **High fidelity**: Achieves high accuracy in reproducing the actual eclipse.
- **Integration with Taylor-based integrator**: they can be used as event detection functions inside Taylor propagators, enabling differentiable and reliable event detection.

The approach unlocks **differentiable eclipse description** for low-thrust transfers, station-keeping, and proximity operations around small bodies, where accurate shadowing prediction is critical but computationally challenging.

<img width="6600" height="5400" alt="fig" src="https://github.com/user-attachments/assets/cbb4f493-f209-443d-b334-e4ed60d46776" />


## Citation

If you use this work, please cite the following publication:

```bibtex
@article{ACCIARINI2025514,
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
