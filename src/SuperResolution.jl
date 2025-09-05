module SuperResolution


export edsrmodel, EDSRBaseline, EDSRExpanded
export esrgan, ESRGANBaseline, ESRGANExtended

import Flux
import Flux: Chain, SkipConnection, relu, sigmoid, leakyrelu, @layer
import TinyMachines: ConvK3

include("edsr_blocks.jl")
include("esrgan_blocks.jl")

include("edsr.jl")
include("esrgan.jl")


end   # module
