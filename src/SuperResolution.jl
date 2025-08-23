module SuperResolution


export edsrmodel, EDSRBaseline, EDSRExpanded

import Flux
import Flux: Chain, SkipConnection, relu, sigmoid, @layer
import TinyMachines: ConvK3

include("blocks.jl")
include("edsr.jl")
# include("esrgan.jl")


end   # module
