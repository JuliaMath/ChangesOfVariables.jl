module ChainRulesCoreExt

using ChainRulesCore

import ChangesOfVariables: _with_ladj_on_mapped

# Need to use a type for this, type inference fails when using a pullback
# closure over YLT in the rrule, resulting in bad performance:
struct WithLadjOnMappedPullback{YLT} <: Function end
function (::WithLadjOnMappedPullback{YLT})(thunked_ΔΩ) where YLT
    ys, ladj = unthunk(thunked_ΔΩ)
    return NoTangent(), NoTangent(), map(y -> Tangent{YLT}(y, ladj), ys)
end

function ChainRulesCore.rrule(::typeof(_with_ladj_on_mapped), map_or_bc::F, y_with_ladj) where {F<:Union{typeof(map),typeof(broadcast)}}
    YLT = eltype(y_with_ladj)
    return _with_ladj_on_mapped(map_or_bc, y_with_ladj), WithLadjOnMappedPullback{YLT}()
end

end
