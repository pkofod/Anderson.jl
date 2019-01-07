module Anderson
using LinearAlgebra, Parameters
"""
    qrdelete!(Q, R)

Delete the left-most column of F = Q*R by updating Q and R.
The right-most column of Q and last column/last row of R is
invalid on exit.
"""
# TODO should only run up to m_eff!
function qrdelete!(Q::AbstractMatrix{T}, R) where T
    n, m = size(Q)
    for i = 1:m-1
        temp = sqrt(R[i, i+1]^2 + R[i+1, i+1]^2)
        c = R[i, i+1]/temp
        s = R[i+1, i+1]/temp
        R[i, i+1] = temp
        R[i+1, i+1] = T(0)
        if i < m-1
            for j = i+2:m
                temp = c*R[i, j] + s*R[i+1, j]
                R[i+1, j] = -s*R[i, j] + c*R[i+1, j]
                R[i, j] = temp
            end
        end

        for l = 1:n
            temp = c*Q[l, i] + s*Q[l, i+1]
            Q[l, i+1] = -s*Q[l, i] + c*Q[l, i+1]
            Q[l, i] = temp
        end
    end
    # my addition: move it up into the corner
    for j = 2:m
        for i = 1:m-1
            R[i, j-1] = R[i, j]
        end
    end
end

@with_kw struct AndersonCache{Tm, Tv, Tmcv, Tmgv, Tmrv, Tmqv, Tvγ, Tvmc}
    G::Tm
    Gv::Tmgv
    R::Tm
    Rv::Tmrv
    Rcv::Tmcv
    Q::Tm
    Qv::Tmqv
    fold::Tv
    fcur::Tv
    Δf::Tv
    gold::Tv
    gcur::Tv
    Δg::Tv
    m::Int
    γ::Tvγ
    m_eff_cache::Tvmc
    x_cache::Tv
end

function AndersonCache(x0, m)
    n = length(x0)
    m_corrected = min(n, m)
    G = similar(x0, n, m_corrected)
    Gv = [view(G, :, 1:i) for i = 1:m_corrected]
    R = similar(x0, m_corrected, m_corrected)*0
    Rv = [view(R, 1:i, 1:i) for i = 1:m_corrected]
    Rcv = [similar(x0, i, i) for i = 1:m_corrected]
    Q = similar(x0, n, m_corrected)
    Qv = [view(Q, :, 1:i) for i = 1:m_corrected]
    fold = similar(x0)
    fcur = similar(x0)
    Δf = similar(fcur)
    gold = similar(x0)
    gcur = similar(x0)
    Δg = similar(gcur)
    γ = similar(x0, m)
    γv = [view(γ, 1:i) for i = 1:m_corrected]

    m_eff_cache = similar(x0, m_corrected)
    m_cache = [view(m_eff_cache, 1:i) for i = 1:m_corrected]
    x_cache = similar(x0)

    AndersonCache(G, Gv, R, Rv, Rcv, Q, Qv, fold, fcur, Δf, gold, gcur, Δg, m_corrected, γv, m_cache, x_cache)
end
# if some term converges exactly, you can get lapack errors (try to set an x to the true value)
function anderson(g, x0; itermax=1000, m=10, delay=0, mutate_x0=false, droptol=1e10, beta=nothing)
    if mutate_x0
        x = x0
    else
        x = copy(x0)
    end
    cache = AndersonCache(x0, m)
    anderson!(g, x, cache; itermax=itermax, delay=delay, droptol=droptol, beta=beta)
end

function function_iteration!(g, x::AbstractArray{T}, iter,
                            tol=sqrt(eps(T)), gcur=similar(x), fcur=similar(x)) where T
    for k = 1:iter
#        gcur .= g(x)
        g(gcur, x)
        fcur .= gcur .- x
        x .= gcur
        if norm(fcur, Inf) < tol
            return x, true
        end
    end
    x, false
end
function anderson!(g, x::AbstractArray{T}, cache; itermax = 1000, delay=0, tol=sqrt(eps(T)), droptol=T(1e10), beta=nothing) where T
    @unpack G, Gv, R, Rv, Rcv, Q, Qv, fold, fcur, Δf, gold, gcur, Δg, m, γ, m_eff_cache, x_cache = cache
    function_iterations = delay+1 # always do one function iteration to set up gold and fold
    x, function_converged = function_iteration!(g, x, function_iterations, tol, gcur, fcur)
    if function_converged
        println("Converged before acceleration began")
        return x, true, 0
    end
    # k is now 0
    if m == 0
        x, function_converged = function_iteration!(g, x, itermax, tol, gcur, fcur)
        if function_converged
            println("Converged without acceleration (m=0).")
            return x, true, 0
        else
            return x, false, 0
        end
    end

    # Since we always do one function iteration we can store the current values
    # in gold and fold. We've already checked for convergence.
    gold .= gcur
    fold .= fcur
    # m_eff is the current history length (<= cache)
    m_eff = 0
    verbose = false
    # start acceleration
    for k = 1:itermax
        #gcur .= g(x)
        g(gcur, x)
        fcur .= gcur .- x
        verbose && println()
        verbose && println("$k) Residual: ", norm(fcur, Inf))
        if norm(fcur, Inf) < tol
            return x, true, k
        end
        Δf .= fcur .- fold
        Δg .= gcur .- gold

        # now that we have the difference, we can update the cache arrays
        # for f and g
        copyto!(fold, fcur)
        copyto!(gold, gcur)

        # if effective memory counter is below the maximum
        # memory length, use the iteration index. Otherwise, update from the
        # right.
        m_eff = m_eff + 1
        if m_eff <= m
            G[:, m_eff] = Δg
        else
            for i = 1:size(G, 1)
                for j = 2:m
                    G[i, j-1] = G[i, j]
                end
            end
            G[:, end] = Δg
        end
#        show(G)
        # increment effective memory counter

        # if m_eff == 1 simply update R[1, 1] and Q[:, 1], else we need to
        # loop over rows in R[:, m_eff] and columns in Q. Potentially, we
        # need to delete a column from the right in Q and the last row/column
        # in R first, if the effective memory counter is larger than the
        # maximum memory length.
        if m_eff > 1
            if m_eff > m
                m_eff = m_eff - 1
                qrdelete!(Qv[m_eff], Rv[m_eff])
            end
            for i = 1:m_eff-1
                # R[i, m_eff] = dot(Q[:, i], Δf)
                R[i, m_eff] = T(0)
                for j = 1:length(Δf)
                    R[i, m_eff] += Q[j, i] * Δf[j]
                end
#                Δf .= Δf .- R[i, m_eff].*Q[:, i]
                for j = 1:length(Δf)
                    Δf[j] = Δf[j] - R[i, m_eff]*Q[j, i]
                end
            end
        end
       R[m_eff, m_eff] = norm(Δf, 2)
        # R[m_eff, m_eff] = T(0)
        # for j = 1:length(Δf)
        #     R[m_eff, m_eff] += Δf[j]^2
        # end
        # R[m_eff, m_eff] = sqrt(R[m_eff, m_eff])
        Q[:, m_eff] .= Δf./R[m_eff, m_eff]


        if !isa(droptol, Nothing)
            condR = cond(Rv[m_eff])
            while condR > droptol && m_eff > 1
                qrdelete!(Qv[m_eff], Rv[m_eff])
                for j = 2:m_eff
                    for i = 1:size(G, 1)
                        G[i, j-1] = G[i, j]
                    end
                end
                m_eff = m_eff - 1
                condR = cond(Rv[m_eff])
            end
        end

        Rcv[m_eff] .= Rv[m_eff]
        #    γ[1:m_eff] .= Rv[m_eff]\Qv[m_eff]'*fcur
        F = qr!(Rcv[m_eff])
        ldiv!(γ[m_eff], F, mul!(m_eff_cache[m_eff], Qv[m_eff]', fcur))
        x .= gcur .- mul!(x_cache, Gv[m_eff], γ[m_eff])
        if !isa(beta, Nothing)
#            x .= x .- (1 .- beta)*(fcur .- Q[m_eff]*R[m_eff]*γ[m_eff])
            x .= x .- (1 .- beta).*(fcur .- mul!(x_cache, Qv[m_eff], mul!(m_eff_cache[m_eff], Rv[m_eff], γ[m_eff])))
        end
    end
    printstyled("Failure!\n"; color=9)
    x, false, itermax
end

end # module
