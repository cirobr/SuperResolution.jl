struct ResidualDenseBlock
    layers::Chain
end
@layer ResidualDenseBlock

function ResidualDenseBlock(num_features, gc, activation=relu)
    front_layers = [ConvK3(num_features + i*gc, gc, activation) for i in 0:3]  # 4 conv layers
    final_layer  = ConvK3(num_features + 4*gc, num_features)  # final conv layer without activation

    return Chain(front_layers..., final_layer)
end

function (rdb::ResidualDenseBlock)(x)
    residual_scale = 0.2
    x1 = rdb.layers[1](x)
    x2 = rdb.layers[2](cat(x, x1), dims=3)
    x3 = rdb.layers[3](cat(x, x1, x2), dims=3)
    x4 = rdb.layers[4](cat(x, x1, x2, x3), dims=3)
    x5 = rdb.layers[5](cat(x, x1, x2, x3, x4), dims=3)

    return x5 .* residual_scale .+ x
end


struct RRDB
    blocks::Chain
end
@layer RRDB

function RRDB(num_features, gc, activation)
    blocks = [ResidualDenseBlock(num_features, gc, activation) for _ in 1:3]  # 3 RDBs
    return Chain(blocks...)
end

function (rrdb::RRDB)(x)
    residual_scale = 0.2
    y = rrdb.blocks(x)

    return y .* residual_scale .+ x
end
