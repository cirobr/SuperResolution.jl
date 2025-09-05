X = rand(Float32, 64,64,3,1)

model = sr.ResidualDenseBlock(num_features=3, num_layers=8, activation=relu)
y = model(X)
@test size(y) == (64,64,3,1)

model = sr.RRDB(num_features=3, num_layers=8, residual_scale=0.2f0, activation=relu)
y = model(X)
@test size(y) == (64,64,3,1)

model = esrgan(3, 3, num_features=64, num_layers=32, num_rrdb=16, scale=2, activation=relu)
y = model(X)
@test size(y) == (128,128,3,1)

model = esrgan(3, 3, num_features=64, num_layers=32, num_rrdb=16, scale=3, activation=relu)
y = model(X)
@test size(y) == (192,192,3,1)

model = esrgan(3, 3, num_features=64, num_layers=32, num_rrdb=16, scale=4, activation=relu)
y = model(X)
@test size(y) == (256,256,3,1)

model = ESRGANBaseline()
y = model(X)
@test size(y) == (128,128,3,1)

model = ESRGANExtended()
y = model(X)
@test size(y) == (128,128,3,1)
