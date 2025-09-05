X = rand(Float32, 64,64,3,1)

model = edsr(3, 3, num_layers=16, num_features=64, scale=2, residual_scale=1.0f0, activation=relu)
y = model(X)
@test size(y) == (128,128,3,1)

model = edsr(3, 3, num_layers=16, num_features=64, scale=3, residual_scale=1.0f0, activation=relu)
y = model(X)
@test size(y) == (192,192,3,1)

model = edsr(3, 3, num_layers=16, num_features=64, scale=4, residual_scale=1.0f0, activation=relu)
y = model(X)
@test size(y) == (256,256,3,1)

model = EDSRBaseline(scale=2)
y = model(X)
@test size(y) == (128,128,3,1)

model = EDSRExpanded(scale=2)
y = model(X)
@test size(y) == (128,128,3,1)
