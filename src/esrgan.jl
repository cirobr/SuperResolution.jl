# Residual in Residual Dense Block (RRDB)
function RRDB(ch_in::Int;
    num_features::Int,
    num_layers::Int,
    residual_scale::Float32,
    activation::Function
)
    rb = ResidualBlock(ch_in, ch_in,
            num_features   = num_features,
            activation     = activation,
            residual_scale = residual_scale)
    # skip_rb = SkipConnection(rb, +)
    rb_vector = [rb for _ in 1:num_layers]
    chain = Chain(rb_vector..., x -> x .* residual_scale)

    return SkipConnection(chain, +)
end

# ESRGAN constructor
function esrganmodel(
    ch_in::Int=3,
    ch_out::Int=3;
    num_features::Int=64,
    num_layers::Int=16,
    residual_scale::Float32=0.2f0,
    activation::Function=relu
    )

    @assert num_features % 4 == 0 || error("num_features must be divisible by 4 for PixelShuffle")
    
    head = ConvK3(ch_in, num_features)

    body = RRDB(
        num_features,
        num_features=num_features,
        num_layers=num_layers,
        residual_scale=residual_scale,
        activation=activation
    )

    # 4X upscale
    # upscale = Chain(
    #     ConvK3(num_features, num_features, activation),
    #     Flux.PixelShuffle(2),  # First 2X upscaling, features divided by 2^2
    #     ConvK3(num_features ÷ 4, num_features, activation),
    #     Flux.PixelShuffle(2)   # Second 2X upscaling, features divided by 2^2
    # )
    upscale = Upsample4X(num_features, activation=activation)

    # tail = Chain(
    #     ConvK3(num_features ÷ 4, num_features ÷ 4, activation),
    #     ConvK3(num_features ÷ 4, ch_out, sigmoid)   # sigmoid output activation
    # )
    tail = Chain(
        ConvK3(num_features, num_features, activation),
        ConvK3(num_features, ch_out, sigmoid)   # sigmoid output activation
    )

    return Chain(h=head, b=body, up=upscale, t=tail)
end

ESRGAN() = esrganmodel(
    3,
    3,
    num_features=64,
    num_layers=16,
    residual_scale=0.2f0,
    activation=leakyrelu
)