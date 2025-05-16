# java-fusion
Denoising Diffusion Probabilistic Model (DDPM) in Java using threads and ND4J. This repo contains a basic from-scratch implementation of the DDPM process, with a focus on clarity, simplicity, and using core ML concepts without hiding behind high-level libraries.

This is not meant to be fast or GPU-optimized — it’s meant to be understandable.

---

## What’s this about?

The idea behind a DDPM is to *model data as a noisy process*. We take an image (or tensor), gradually corrupt it with Gaussian noise, and then train a model to reverse that process step-by-step — so that it can turn pure noise back into something meaningful.

This repo does that in Java. From scratch. With threads.

---

## How it works (in this repo)

### The Forward Process (q)
We take a clean sample `x₀`, and add noise step-by-step:

xₜ = √αₜ * x₀ + √(1 - αₜ) * ε


Where:
- `αₜ` is a scalar that defines how much of the original sample to keep at time `t`
- `ε` is Gaussian noise
- We do this for `T` steps, getting progressively noisier versions

We implement this using ND4J arrays and Java random number generators. The `α` schedule is precalculated and stored in memory.

### The Reverse Process (p)
We train a neural net to estimate the noise `ε` added at each step, given a noisy sample `xₜ` and the time step `t`. Once trained, the network lets us sample like this:

xₜ₋₁ = (xₜ - √(1 - αₜ) * ε_θ(xₜ, t)) / √αₜ + some noise


This is implemented as an iterative denoising loop. We start from random noise and denoise it step-by-step using the model.

### The Model (U-Net-ish)
The network is a small U-Net style convolutional network. It has:
- A downsampling block (2D convs with ReLU)
- A bottleneck
- An upsampling block (transposed convs)

Each block is manually wired up using ND4J matrix operations — no fancy autograd here. We apply convolution using custom sliding windows (a bit expensive but easy to follow). Skip connections can optionally be added.

---

## Multithreading

To make things a bit faster (and show off Java's concurrency), the image batch processing is parallelized. Each thread handles one sample during training or sampling. We use `ExecutorService` to manage threads, and ND4J handles the array math per thread.

This isn’t the fastest way, but it’s clean and works.

---

## Math Recap (Lightweight)

- Forward process: `q(xₜ | x₀)` is a known Gaussian process
- Reverse process: we learn `ε_θ(xₜ, t)` to predict noise
- Loss: we train the model to minimize:

L = E[ || ε - ε_θ(xₜ, t) ||² ]



This is just L2 loss between actual noise and predicted noise, given the noisy input.

---

## Why Java?

Because it’s not Python, and that’s the point. If you know how to use arrays and loops, you can follow this project. It's all transparent and close to the math. Plus, ND4J lets you handle N-dimensional arrays like in NumPy, so it's a great fit.

---

## What’s missing

- No fancy features like classifier-free guidance
- No GPU acceleration (CPU only)
- No checkpoint saving/loading
- Training images are synthetic tensors, no dataset loader yet

---

# Bibliography

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.  
  [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
