# ESRGAN Model
function ESRGAN(ch_in::Int=3, ch_out::Int=3;
                num_features::Int=64,
                gc::Int=32,
                num_rrdb::Int=23,
                scale::Int=2,                    # upscale factor
                activation::Function=relu
)
    head = ConvK3(ch_in, num_features)

    rrdb_blocks = [RRDB(num_features, gc, activation) for _ in 1:num_rrdb]
    body_layers = Chain(rrdb_blocks..., ConvK3(num_features, num_features))
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
