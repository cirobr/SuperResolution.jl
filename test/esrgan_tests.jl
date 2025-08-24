X = rand(Float32, 64,64,3,1)
model = esrganmodel(3,3)
y = model(X)
@test size(y) == (256,256,3,1)

model = ESRGAN()
y = model(X)
@test size(y) == (256,256,3,1)
