# https://arxiv.org/pdf/1809.00219

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
    skip_rb = SkipConnection(rb, +)
    skip_rbs = [skip_rb for _ in 1:num_layers]
    chain = Chain(skip_rbs..., x -> x .* residual_scale)

    return SkipConnection(chain, +)
end

# constructor
function esrganmodel(
    ch_in::Int=3,
    ch_out::Int=3;
    num_features::Int=64,
    num_layers::Int=16,
    residual_scale::Float32=0.2f0,
    activation::Function=relu
    )

    @assert num_features % 4 == 0 || error("num_features must be divisible by 4 for PixelShuffle")
    
    head = ConvK3(ch_in, num_features, activation)

    basic_block = RRDB(
        num_features,
        num_features=num_features,
        num_layers=num_layers,
        residual_scale=residual_scale,
        activation=activation
    )
    basic_blocks = [basic_block for _ in 1:num_layers]
    bd   = Chain(basic_blocks...)
    body = Chain(SkipConnection(bd, +), ConvK3(num_features, num_features))

    upscale = Upsample4X(num_features, activation=activation)

    tail = Chain(
        ConvK3(num_features, num_features, activation),
        ConvK3(num_features, ch_out, sigmoid)
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