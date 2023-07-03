module ChangesOfVariablesInverseFunctionsExt

using ChangesOfVariables
using InverseFunctions


struct InverseFunctionWithLADJ{InvF,LADJF} <: Function
    inv_f::InvF
    ladjf::LADJF
end
InverseFunctionWithLADJ(::Type{InvF}, ladjf::LADJF) where {InvF,LADJF} = InverseFunctionWithLADJ{Type{InvF},LADJF}(InvF,ladjf)
InverseFunctionWithLADJ(inv_f::InvF, ::Type{LADJF}) where {InvF,LADJF} = InverseFunctionWithLADJ{InvF,Type{LADJF}}(inv_f,LADJF)
InverseFunctionWithLADJ(::Type{InvF}, ::Type{LADJF}) where {InvF,LADJF} = InverseFunctionWithLADJ{Type{InvF},Type{LADJF}}(InvF,LADJF)

(f::InverseFunctionWithLADJ)(y) = f.inv_f(y)

function ChangesOfVariables.with_logabsdet_jacobian(f::InverseFunctionWithLADJ, y)
    x = f.inv_f(y)
    return x, -f.ladjf(x)
end

InverseFunctions.inverse(f::ChangesOfVariables.FunctionWithLADJ) = InverseFunctionWithLADJ(inverse(f.f), f.ladjf)
InverseFunctions.inverse(f::InverseFunctionWithLADJ) = ChangesOfVariables.FunctionWithLADJ(inverse(f.inv_f), f.ladjf)


@static if isdefined(InverseFunctions, :FunctionWithInverse)
    function ChangesOfVariables.with_logabsdet_jacobian(f::InverseFunctions.FunctionWithInverse, x)
        ChangesOfVariables.with_logabsdet_jacobian(f.f, x)
    end
end

end # module ChangesOfVariablesInverseFunctionsExt
