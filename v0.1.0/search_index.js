var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"DocTestSetup  = quote\n    using ChangesOfVariables\nend","category":"page"},{"location":"api/#Modules","page":"API","title":"Modules","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:module]","category":"page"},{"location":"api/#Types-and-constants","page":"API","title":"Types and constants","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:type, :constant]","category":"page"},{"location":"api/#Functions-and-macros","page":"API","title":"Functions and macros","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:macro, :function]","category":"page"},{"location":"api/#Documentation","page":"API","title":"Documentation","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [ChangesOfVariables]\nOrder = [:module, :type, :constant, :macro, :function]","category":"page"},{"location":"api/#ChangesOfVariables.ChangesOfVariables","page":"API","title":"ChangesOfVariables.ChangesOfVariables","text":"ChangesOfVariables\n\nLightweight package that defines functionality to calculate volume element changes for functions that perform a change of variables (like coordinate transformations).\n\n\n\n\n\n","category":"module"},{"location":"api/#ChangesOfVariables.with_logabsdet_jacobian","page":"API","title":"ChangesOfVariables.with_logabsdet_jacobian","text":"with_logabsdet_jacobian(f, x)\n\nComputes both the transformed value of x under the transformation f and the logarithm of the volume element.\n\nFor (y, ladj) = with_logabsdet_jacobian(f, x), the following must hold true:\n\ny == f(x)\nladj is the log(abs(det(jacobian(f, x))))\n\nwith_logabsdet_jacobian comes with support for broadcasted/mapped functions (via Base.Fix1) and (Julia >=v1.6 only) ComposedFunction.\n\nExample:\n\nfoo(x) = inv(exp(-x) + 1)\n\nfunction ChangesOfVariables.with_logabsdet_jacobian(::typeof(foo), x)\n    y = foo(x)\n    ladj = -x + 2 * log(y)\n    (y, ladj)\nend\n\nx = 4.2\ny, ladj_y = with_logabsdet_jacobian(foo, x)\n\nX = rand(10)\nbroadcasted_foo = Base.Fix1(broadcast, foo)\nY, ladj_Y = with_logabsdet_jacobian(broadcasted_foo, X)\n\n# Requires Julia >= v1.6:\nz, ladj_z = with_logabsdet_jacobian(log ∘ foo, x)\nz == log(foo(x))\nladj_z == ladj_y + with_logabsdet_jacobian(log, y)[2]\n\n\n\n\n\n","category":"function"},{"location":"LICENSE/#LICENSE","page":"LICENSE","title":"LICENSE","text":"","category":"section"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"using Markdown\nMarkdown.parse_file(joinpath(@__DIR__, \"..\", \"..\", \"LICENSE.md\"))","category":"page"},{"location":"#ChangesOfVariables.jl","page":"Home","title":"ChangesOfVariables.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package defines the function with_logabsdet_jacobian. (y, ladj) = with_logabsdet_jacobian(f, x) computes both the transformed value of x under the transformation f and the logarithm of the volume element.","category":"page"},{"location":"","page":"Home","title":"Home","text":"with_logabsdet_jacobian supports mapped/broadcasted functions (via Base.Fix1) and (on Julia >=v1.6) function composition.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Implementations of with_logabsdet_jacobian(f) for identity, inv, adjoint and transpose as well as for exp, log, exp2, log2, exp10, log10, expm1 and log1p are included.","category":"page"}]
}