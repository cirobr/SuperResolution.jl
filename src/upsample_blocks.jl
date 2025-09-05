function UpsampleBlock(channels::Int; scale::Int, activation::Function)
    @assert scale ∈ (2:3) || error("Scale must be 2 or 3")
    return Chain(
        ConvK3(channels, channels * scale^2),
        Flux.PixelShuffle(scale),
        activation
    )
end


Upsample2X(channels::Int; activation::Function) = 
    UpsampleBlock(channels, scale=2, activation=activation)

Upsample3X(channels::Int; activation::Function) = 
    UpsampleBlock(channels, scale=3, activation=activation)

Upsample4X(channels::Int; activation::Function) =
    Chain(
        UpsampleBlock(channels, scale=2, activation=activation),
        UpsampleBlock(channels, scale=2, activation=activation)
    )
