module SuperResolution


export edsrmodel, EDSRBaseline, EDSRExpanded
export esrganmodel, ESRGAN

import Flux
import Flux: Chain, SkipConnection, relu, sigmoid, leakyrelu, @layer
import TinyMachines: ConvK3

include("blocks.jl")
include("edsr.jl")
include("esrgan.jl")


end   # module
