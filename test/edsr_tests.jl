X = rand(Float32, 64,64,3,1)

model = edsrmodel(3, 3, scale=2)
y = model(X)
@test size(y) == (128,128,3,1)

model = edsrmodel(3, 3, scale=3)
y = model(X)
@test size(y) == (192,192,3,1)

model = edsrmodel(3, 3, scale=4)
y = model(X)
@test size(y) == (256,256,3,1)
