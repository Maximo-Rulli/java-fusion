# java-fusion  
Denoising Diffusion Probabilistic Model (DDPM) trained on MNIST, implemented in Java for sampling, using threads and ND4J. The training is done externally in Python — this repo focuses on sampling: given a trained model that predicts noise, we reconstruct data from pure noise in a step-wise fashion. The network runs in Java, no autodiff, no frameworks, just raw arrays and control.

---

## What’s a DDPM?

The DDPM is a generative model that turns noise into structure. The process is built from two parts:

- **Forward process (diffusion)**: gradually add Gaussian noise to data over `T` time steps until it becomes indistinguishable from random noise.
- **Reverse process (denoising)**: train a model to remove that noise one step at a time.

The magic happens because each step is simple — linear, Gaussian — and we can analytically describe the distribution at each point. So even though the full path from noise to structure is complex, we approximate it piece by piece.

The result: a model that can sample complex data (like images) starting from pure noise.

---

## How this is implemented

This Java project handles **only the sampling**. You feed it a trained model (converted to ONNX or raw weights), and it walks backward from noise to signal.

### ND4J Arrays

ND4J is used for handling tensor-like structures. Arrays are explicitly shaped, operated with manual broadcasting and reshaping where necessary. No magic.

### Sampling Loop

We follow this schedule:

```

x₀ ← ε  (random normal noise)

for t = T down to 1:
ε_θ ← predict(xₜ, t)
xₜ₋₁ ← (1/√αₜ) * (xₜ - √(1 - αₜ) * ε_θ) + σₜ * z

```

Where `αₜ` and `σₜ` are precomputed scalars based on a linear or cosine schedule, and `z` is Gaussian noise (zeroed at the last step).

This loop is implemented in plain Java, using threads for parallel execution across batches.

---

## The Network

The model is a U-Net. Structurally simple, conceptually deep:

- **Downsampling path**: A sequence of 2D convolutions + non-linearities that progressively compress spatial information into lower-dimensional representations.
- **Bottleneck**: The dense representation — the point where all local context is merged.
- **Upsampling path**: A sequence of **transposed convolutions** that increase the spatial dimensions, aiming to reconstruct the original structure.

Skip connections bridge matching levels in downsampling and upsampling, allowing early features to inform reconstruction directly.

#### What is a convolution?

Imagine a small matrix (kernel) sliding across a larger matrix (image), performing an element-wise multiplication and summing the result. This picks up patterns — edges, textures, gradients. The weights of the kernel are what the model learns.

#### What is a transposed convolution?

It’s not the inverse of a convolution, but it *reverses* the effect in terms of size. Instead of reducing spatial dimensions, it expands them. It’s how you go from compact latent representations back to full-size outputs. Think of it like unpooling with learned parameters.

---

## Multithreading

The sampling step is embarrassingly parallel — each image can be sampled independently. This project uses Java’s `ExecutorService` to run samples concurrently across CPU threads. It's straightforward, but makes use of what Java gives you.

---

## Why Java?

The point is transparency. Java forces you to spell everything out. There are no implicit graphs, no hidden gradients, no automatic shape inference. Every slice, broadcast, and reshape is your responsibility.

It’s not for convenience. It’s for clarity — and maybe just a little bit of stubbornness.

---

## Requirements

- Java 17+
- Maven
- Trained model weights (included in repo)

---

## Citation

```bibtex
@misc{ho2020denoisingdiffusionprobabilisticmodels,
  title={Denoising Diffusion Probabilistic Models}, 
  author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
  year={2020},
  eprint={2006.11239},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2006.11239}
}

@misc{ronneberger2015unetconvolutionalnetworksbiomedical,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
  author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
  year={2015},
  eprint={1505.04597},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/1505.04597}
}
