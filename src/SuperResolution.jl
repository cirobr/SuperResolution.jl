module SuperResolution


export edsrmodel, EDSRbaseline, EDSRExpanded

import Flux
import Flux: Chain, SkipConnection, relu, sigmoid, @layer
import TinyMachines: ConvK3

include("edsr.jl")


end   # module
