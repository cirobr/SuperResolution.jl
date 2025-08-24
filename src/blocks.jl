function ResidualBlock(ch_in::Int, ch_out::Int;
    num_features::Int,
    activation::Function,
    residual_scale::Float32,
)
    chain = Chain(
        ConvK3(ch_in, num_features, activation),
        ConvK3(num_features, ch_out),
        x -> x .* residual_scale
    )

    return SkipConnection(chain, +)
end


function UpsampleBlock(channels::Int; scale::Int, activation::Function)
    @assert scale ∈ (2:3) || error("Scale must be 2 or 3")
    return Chain(
        ConvK3(channels, channels * scale^2),
        Flux.PixelShuffle(scale),
        activation
    )
end
