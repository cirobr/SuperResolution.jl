# ESRGAN Model
function esrgan(
                ch_in::Int, ch_out::Int;     # input/output channels
                num_features::Int,           # RRDB input feature channels
                num_layers::Int,             # growth channels for RDB
                num_rrdb::Int,               # number of RRDB blocks
                scale::Int,                  # upscale factor
                activation::Function         # hidden activation function
)
    @assert scale ∈ (2, 3, 4) || error("Scale must be 2, 3, or 4")

    # structure
    head = ConvK3(ch_in, num_features)

    basic_block = RRDB(num_features=num_features, num_layers=num_layers, activation=activation)
    basic_blocks = [basic_block for _ in 1:num_rrdb]
    body_layers = Chain(basic_blocks..., ConvK3(num_features, num_features))
    body = SkipConnection(body_layers, +)

    upsample = scale == 2 ? Upsample2X(num_features, activation=activation) :
               scale == 3 ? Upsample3X(num_features, activation=activation) :
                            Upsample4X(num_features, activation=activation)

    tail = Chain(
        ConvK3(num_features, num_features, activation),
        ConvK3(num_features, ch_out, sigmoid)
    )

    return Chain(head, body, upsample, tail)
end

ESRGANBaseline(;scale=2) = esrgan(3, 3,
                                num_features=64,
                                num_layers=32,
                                num_rrdb=16,
                                scale=scale,
                                activation=leakyrelu
)

ESRGANExtended(;scale=2) = esrgan(3, 3,
                                num_features=64,
                                num_layers=32,
                                num_rrdb=23,
                                scale=scale,
                                activation=leakyrelu
)
