struct ResidualDenseBlock
    layers::Chain
end
@layer ResidualDenseBlock

function ResidualDenseBlock( ;
            num_features::Int,       # input feature channels
            num_layers::Int,         # growth channels for each conv layer
            activation::Function     # activation function
)
    front_layers = [ConvK3(num_features + i*num_layers, num_layers, activation) for i in 0:3]  # 4 conv layers
    final_layer  = ConvK3(num_features + 4*num_layers, num_features)  # final conv layer without activation
    chain = Chain(front_layers..., final_layer)

    return ResidualDenseBlock(chain)
end

function (rdb::ResidualDenseBlock)(x)
    residual_scale = 0.2f0
    x1 = rdb.layers[1](x)
    x2 = rdb.layers[2](cat(x, x1, dims=3))
    x3 = rdb.layers[3](cat(x, x1, x2, dims=3))
    x4 = rdb.layers[4](cat(x, x1, x2, x3, dims=3))
    x5 = rdb.layers[5](cat(x, x1, x2, x3, x4, dims=3))

    return x5 .* residual_scale .+ x
end


function RRDB( ;
    num_features::Int,              # input feature channels
    num_layers::Int,                # growth channels for RDB
    activation::Function,           # activation function
    residual_scale::Float32=0.2f0   # residual scale factor (paper's default)
)
    blocks = [ResidualDenseBlock(num_features=num_features, num_layers=num_layers, activation=activation) for _ in 1:3]
    chain = Chain(blocks..., x -> x .* residual_scale)   # scale the output

    return SkipConnection(chain, +)
end
