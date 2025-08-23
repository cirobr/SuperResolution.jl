# https://arxiv.org/abs/1707.02921

struct ResidualBlock
    chain::Chain
    residual_scale
end
@layer ResidualBlock trainable=(chain)

function ResidualBlock(channels::Int; residual_scale)
    chain = Chain(
        ConvK3(channels, channels, relu),
        ConvK3(channels, channels)
    )

    return ResidualBlock(chain, residual_scale)
end

function (m::ResidualBlock)(x)
    return x .+ m.residual_scale .* m.chain(x)
end



function UpsampleBlock(channels::Int, scale::Int)   # check for scale 2,3,4
    return Chain(
        ConvK3(channels, channels * scale^2),
        Flux.PixelShuffle(scale)
    )
end



# constructor
function edsrmodel(
    ch_in::Int=3,
    ch_out::Int=3;
    B::Int=16,             # depth (number of layers) of the body
    F::Int=64,             # width (number of feature channels)
    scale::Int=2,          # upscale factor
    residual_scale=1.0f0   # residual scaling factor
)
    @assert scale in (2, 3, 4) || error("Scale must be 2, 3, or 4")

    head = ConvK3(ch_in, F)
    tail = ConvK3(F, ch_out)

    rbs  = [ResidualBlock(F, residual_scale=residual_scale) for _ in 1:B]
    bd   = Chain(rbs...)
    body = SkipConnection(bd, +)

    upsample = UpsampleBlock(F, scale)

    return Chain(head, body, upsample, tail, x ->sigmoid.(x))
end

# Baseline: B=16, F=64, residual_scale=1
EDSRBaseline(scale::Int=2) = edsrmodel(3, 3, B=16, F=64, scale=scale, residual_scale=1.0f0)

# Expanded: B=32, F=256, residual_scale=0.1
EDSRExpanded(scale::Int=2) = edsrmodel(3, 3, B=32, F=256, scale=scale, residual_scale=0.1f0)
