# https://arxiv.org/abs/1707.02921

# constructor
function edsrmodel(
    ch_in::Int=3,                    # input channels
    ch_out::Int=3;                   # output channels
    num_layers::Int=16,              # depth (number of layers) of the body (B in the article)
    num_features::Int=64,            # number of hidden feature channels (F in the article)
    scale::Int=2,                    # upscale factor
    residual_scale::Float32=1.0f0,   # residual scale factor
    activation::Function=relu,       # hidden activation function
)
    @assert scale ∈ (2, 3, 4) || error("Scale must be 2, 3, or 4")

    # structure
    head = ConvK3(ch_in, num_features, activation)

    basic_block = ResidualBlock(
        num_features,
        num_features,
        num_features=num_features,
        activation=activation,
        residual_scale=residual_scale
    )
    basic_blocks = [basic_block for _ in 1:num_layers]
    body_layers  = Chain(basic_blocks...)
    body = Chain(SkipConnection(body_layers, +), ConvK3(num_features, num_features))

    upsample = scale == 2 ? Upsample2X(num_features, activation=activation) :
               scale == 3 ? Upsample3X(num_features, activation=activation) :
                            Upsample4X(num_features, activation=activation)

    tail = ConvK3(num_features, ch_out, sigmoid)   # sigmoid output activation

    return Chain(h=head, b=body, up=upsample, t=tail)
end

# Baseline: B=16, F=64, residual_scale=1
EDSRBaseline(;scale::Int=2) = edsrmodel(3, 3,
                                        num_layers=16,
                                        num_features=64,
                                        scale=scale,
                                        residual_scale=1.0f0,
                                        activation=leakyrelu
)

# Expanded: B=32, F=256, residual_scale=0.1
EDSRExpanded(;scale::Int=2) = edsrmodel(3, 3,
                                        num_layers=32,
                                        num_features=256,
                                        scale=scale,
                                        residual_scale=0.1f0,
                                        activation=leakyrelu
)
