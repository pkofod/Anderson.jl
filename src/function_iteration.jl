function function_iteration!(g, x::AbstractArray{T}, iter,
                            tol=sqrt(eps(T)), gcur=similar(x), fcur=similar(x)) where T
    for k = 1:iter
        g(gcur, x)
        fcur .= gcur .- x
        x .= gcur
        if norm(fcur, Inf) < tol
            return x, true
        end
    end
    x, false
end
function function_iteration(g, x::AbstractArray{T}, iter,
                            tol=sqrt(eps(T))) where T
    for k = 1:iter
        gcur = g(x)
        fcur = gcur - x
        x = gcur
        if norm(fcur, Inf) < tol
            return x, true
        end
    end
    x, false
end
