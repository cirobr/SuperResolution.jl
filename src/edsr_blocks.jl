function ResidualBlock(ch_in::Int, ch_out::Int;
    num_features::Int,
    activation::Function,
    residual_scale::Float32,
)
    chain = Chain(
        Conv3(ch_in, num_features, activation),
        Conv3(num_features, ch_out),
        x -> x .* residual_scale
    )

    return SkipConnection(chain, +)
end
