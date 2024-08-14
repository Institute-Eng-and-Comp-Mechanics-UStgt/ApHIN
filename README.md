<!-- PROJECT SHIELDS -->

[//]: # ([![arXiv][arxiv-shield]][arxiv-url])

[//]: # ([![DOI][doi-shield]][doi-url])
[![Documentation Status][docs-shield]][docs-url]
[![MIT License][license-shield]][license-url]

# ApHIN - Autoencoder-based port-Hamiltonian Identification Networks
A data-driven framework for the identification of latent port-Hamiltonian systems [1].

![ApHIN](https://github.com/user-attachments/assets/0764f063-ced4-4b8e-af84-a7772c7d5c30)

## Abstract
Conventional physics-based modeling techniques involve high effort, e.g.~time and expert knowledge, while data-driven methods often lack interpretability, structure, and sometimes reliability. To mitigate this, we present a data-driven system identification framework that derives models in the port-Hamiltonian (pH) formulation. 
This formulation is suitable for multi-physical systems while guaranteeing the useful system theoretical properties of passivity and stability. 

Our framework combines linear and nonlinear reduction with structured, physics-motivated system identification. 
In this process, high-dimensional state data obtained from possibly nonlinear systems serves as the input for an autoencoder, which then performs two tasks: (i) nonlinearly transforming and (ii) reducing this data onto a low-dimensional manifold. In the resulting latent space, a pH system is identified by considering the unknown matrix entries as weights of a neural network. The matrices strongly satisfy the pH matrix properties through Cholesky factorizations. In a joint optimization process over the loss term, the pH matrices are adjusted to match the dynamics observed by the data, while defining a linear pH system in the latent space per construction.
The learned, low-dimensional pH system can describe even nonlinear systems and is rapidly computable due to its small size.

The method is exemplified by a parametric mass-spring-damper and a nonlinear pendulum example as well as the high-dimensional model of a disc brake with linear thermoelastic behavior.

## Features
This repository implements neural networks that identify linear port-Hamiltonian systems from (potentially high-dimensional) data[1].
* Autoencoders (AEs) for dimensionality reduction
* pH layer to identify system matrices that fullfill the definition of a linear pH system
* pHIN: identify a (parametric) low-dimensional port-Hamiltonian system directly
* ApHIN: identify a (parametric) low-dimensional latent port-Hamiltonian system based on coordinate representations found using an autoencoder
* Examples for the identification of linear pH systems from data
  * One-dimensional mass-spring-damper chain
  * Pendulum
  * discbrake model
  
## Installation

You can either clone the repository and install the package locally or install it directly from PyPI.

### PyPI

```bash
pip install aphin
```

### Local
Clone this repository and install it to your local environment as package using pip:

```bash
git clone https://github.com/Institute-Eng-and-Comp-Mechanics-UStgt/ApHIN.git
cd ApHIN
```
Then you can activate the environment in which you want to install the package, and use pip to perform the installation.
```bash
pip install -e .
```

> :warning: **Please note that you need pip version 24.0 to install the repository in editable mode. Either upgrade pip to the latest version or install it without the ```-e``` argument**

## References

[1] Johannes Rettberg, Jonas Kneifl, Julius Herb, Patrick Buchfink, Jörg Fehr, and Bernard Haasdonk. Data-driven identification of latent port-Hamiltonian systems. Arxiv, 2024.

[2] Volker Mehrmann and Benjamin Unger. Control of port-Hamiltonian differential-algebraic
systems and applications, 2022.

[3] Kathleen Champion, Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. Data-driven
discovery of coordinates and governing equations. Proceedings of the National Academy of
Sciences, 116(45):22445–22451, 2019.

[license-shield]: https://img.shields.io/github/license/Institute-Eng-and-Comp-Mechanics-UStgt/ApHIN.svg
[license-url]: https://github.com/Institute-Eng-and-Comp-Mechanics-UStgt/ApHIN/blob/main/LICENSE
[doi-shield]: https://zenodo.org/badge/DOI/
[doi-url]: https://doi.org/
[arxiv-shield]: https://img.shields.io/badge/arXiv-
[arxiv-url]: https://doi.org/
[docs-url]: https://Institute-Eng-and-Comp-Mechanics-UStgt.github.io/ApHIN
[docs-shield]: https://img.shields.io/badge/docs-online-blue.svg
