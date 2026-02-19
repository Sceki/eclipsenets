# EclipseNETs: Learning Irregular Small Celestial Body Silhouettes

This repository contains the official implementation of the paper:

> [**EclipseNETs: Learning irregular small celestial body silhouettes**](https://www.sciencedirect.com/science/article/abs/pii/S0094576525003443), 
> *Acta Astronautica*, 2025.

## Project Overview

Traditional eclipse modeling for spacecraft trajectory design around asteroids and other irregular small celestial bodies relies on expensive ray-tracing operations against high-fidelity shape models (e.g., polyhedral meshes or spherical harmonics). These methods are computationally prohibitive for real-time applications and large-scale trajectory optimization.

**EclipseNETs** address this challenge by training *differentiable neural networks* to predict irregular **silhouettes** (i.e., the 2D projected shadow of the body as seen from the spacecraft) as a function of spacecraft position and orientation relative to the body's center of mass. The key innovations include:

- **Fully differentiable silhouette prediction**: Enables end-to-end gradient-based optimization of trajectories that account for eclipses.  
- **Fast inference**: Learned models evaluate in microseconds, enabling also much faster integrations of orbital trajectories with eclipse events.  
- **High fidelity**: Achieves high accuracy in reproducing the actual eclipse.
- **Easy integration**: Drop-in replacement for traditional eclipse models in existing trajectory optimization frameworks.

The approach unlocks **differentiable eclipse description** for low-thrust transfers, station-keeping, and proximity operations around small bodies, where accurate shadowing prediction is critical but computationally challenging.

## Citation

If you use this work, please cite the following publication:

```bibtex
@article{acciarini2025eclipsenets,
  title={EclipseNETs: Learning irregular small celestial body silhouettes},
  author={Acciarini, Giacomo and Izzo, Dario and Biscani, Francesco},
  journal={Acta Astronautica},
  year={2025},
  publisher={Elsevier}
}
```
