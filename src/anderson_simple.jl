
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

function simpleanderson(g, x0::AbstractArray{T}; m = 10, itermax = 1000, delay=0, tol=sqrt(eps(T)), droptol=T(1e10), beta=nothing) where T
    if mutate_x0
        x = x0
    else
        x = copy(x0)
    end

    xlen= length(x)
    m = min(length(x), m)

    function_iterations = delay+1 # always do one function iteration to set up gold and fold

    x, function_converged = function_iteration(g, x, function_iterations, tol)
    if function_converged
        println("Converged before acceleration began")
        return x, true, 0
    end
    # k is now 0
    if m == 0
        x, function_converged = function_iteration(g, x, itermax, tol)
        if function_converged
            println("Converged without acceleration (m=0).")
            return x, true, 0
        else
            return x, false, 0
        end
    end

    # Since we always do one function iteration we can store the current values
    # in gold and fold. We've already checked for convergence.
    # m_eff is the current history length (<= cache)
    m_eff = 0
    verbose = false
    # start acceleration
    for k = 1:itermax
        gcur = g(x)
        fcur = gcur - x
        verbose && println()
        verbose && println("$k) Residual: ", norm(fcur, Inf))
        if norm(fcur, Inf) < tol
            return x, true, k
        end
        Δf = fcur - fold
        Δg = gcur - gold

        # now that we have the difference, we can update the cache arrays
        # for f and g
        fold = copy(fcur)
        gold = copy(gcur)

        # if effective memory counter is below the maximum
        # memory length, use the iteration index. Otherwise, update from the
        # right.
        m_eff = m_eff + 1
        if m_eff <= m
            G = [G, Δg]
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
            x .= x .- (1 .- beta).*(fcur .- mul!(x_cache, Qv[m_eff], mul!(m_eff_cache[m_eff], Rv[m_eff], γ[m_eff])))
        end
    end
    printstyled("!!!!!!!!!\n"; color=9)
    printstyled("!Failure!\n"; color=9)
    printstyled("!!!!!!!!!\n"; color=9)
    printstyled("Exited with\n"; color=9)
    printstyled(" * residual inf-norm $(norm(fcur, Inf))\n"; color=9)
    printstyled(" * effective memory length of $m_eff\n"; color=9)
    printstyled("and\n"; color=9)
    printstyled(" - ran for $delay initial function iterations\n"; color=9)
    printstyled(" - ran for $itermax accelerated iterations\n"; color=9)

    x, false, itermax
end
