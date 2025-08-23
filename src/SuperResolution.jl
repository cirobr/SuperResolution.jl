module SuperResolution


export edsrmodel, EDSRBaseline, EDSRExpanded

import Flux
import Flux: Chain, SkipConnection, relu, sigmoid, @layer
import TinyMachines: ConvK3

include("edsr.jl")


end   # module
