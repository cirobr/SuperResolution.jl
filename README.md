![alt text](https://github.com/cirobr/TinyMachines.jl/blob/main/images/logo-name-tm.png?raw=true)

# SuperResolution

[![Build Status](https://github.com/cirobr/SuperResolution.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/SuperResolution.jl/actions/workflows/CI.yml?query=branch%3Amain)

A collection of models aimed at super-resolution applications, written in Julia/Flux.


## EDSR

Reference: Enhanced Deep Residual Networks for Single Image Super-Resolution ([article] (https://arxiv.org/abs/1707.02921)). Authors: Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee.


## Credits
Credits for the original architectures go to the references' authors, as aforementioned.

Credits for the implementations in Julia/Flux go to Ciro B Rosa.
* GitHub: https://github.com/cirobr
* LinkedIn: https://www.linkedin.com/in/cirobrosa/


## Models

```
EDSRBaseline(scale)   # = edsrmodel(3, 3, B=16, F=64, scale=scale, residual_scale=1.0f0)
```

```
EDSRExpanded(scale)   # = edsrmodel(3, 3, B=32, F=256, scale=scale, residual_scale=0.1f0)
```


## Constructors

```
edsrmodel(
    ch_in,                 # number of input channels (default 3)
    ch_out;                # number of output channels (default 3)
    B=16,                  # depth (number of hidden layers) of the body
    F=64,                  # width (number of hidden feature channels)
    scale=2,               # upscale factor
    residual_scale=1.0f0   # residual scaling factor
)
```
