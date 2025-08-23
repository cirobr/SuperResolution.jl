X = rand(Float32, 64,64,3,1)
model = edsrmodel(3, 3, scale=3)
y = model(X)
@test size(y) == (192,192,3,1)
