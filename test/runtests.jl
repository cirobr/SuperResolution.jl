using SuperResolution
using Flux
using Test

@testset "SuperResolution.jl" begin
    include("edsr_tests.jl")
    include("esrgan_tests.jl")
end
