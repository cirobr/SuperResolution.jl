![alt text](https://github.com/cirobr/TinyMachines.jl/blob/main/images/logo-name-tm.png?raw=true)

# SuperResolution

[![Build Status](https://github.com/cirobr/SuperResolution.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/SuperResolution.jl/actions/workflows/CI.yml?query=branch%3Amain)

A collection of models aimed at super-resolution applications, written in Julia/Flux.


## EDSR

Enhanced Deep Residual Networks for Single Image Super-Resolution ([article] (https://arxiv.org/abs/1707.02921)). Authors: Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee.


## ESRGAN

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks ([article] (https://arxiv.org/pdf/1809.00219)). Authors: Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang


## Credits
Credits for the original architectures go to the references' authors, as aforementioned.

Credits for the implementations in Julia/Flux go to Ciro B Rosa.
* GitHub: https://github.com/cirobr
* LinkedIn: https://www.linkedin.com/in/cirobrosa/


## Constructors

```
edsr(
    ch_in::Int, ch_out::Int;     # input/output channels
    num_layers::Int,             # depth (number of layers) of the body (B in the article)
    num_features::Int,           # number of hidden feature channels (F in the article)
    scale::Int,                  # upscale factor
    residual_scale::Float32,     # residual scale factor
    activation::Function,        # hidden activation function
)
```

```
esrgan(
    ch_in::Int, ch_out::Int;     # input/output channels
    num_features::Int,           # RRDB input feature channels
    num_layers::Int,             # growth channels for RDB
    num_rrdb::Int,               # number of RRDB blocks
    scale::Int,                  # upscale factor
    activation::Function         # hidden activation function
)
```
