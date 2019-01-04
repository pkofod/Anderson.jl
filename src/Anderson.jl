module Anderson

struct Anderson
    memory
    delay
    tol
end

"""
    qrdelete!(Q, R)

Delete the left-most column of F = Q*R by updating Q and R.
The right-most column of Q and last column/last row of R is
invalid on exit.
"""
function qrdelete!(Q::AbstractMatrix{T}, R) where T
    n, m = size(Q)
    for i = 1:m-1
        temp = sqrt(R[i, i+1]^2 + R[i+1, i+1]^2^2)
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


# if some term converges exactly, you can get lapack errors (try to set an x to the true value)
function anderson(g, xin; itmax = 1000, mMax = 1)
    x = copy(xin)
    xlen = length(x)
    G = zeros(xlen, mMax)
    R = zeros(mMax, mMax)
    Q = zeros(xlen, mMax)
    fold = similar(x)
    gold = similar(x)
    fval = similar(x)
    gval = similar(x)
    mAA = 0
    for k = 0:itmax

        gval .= g(x)
        fval .= gval - x

        if k > 0
            Δf = fval - fold
            Δg = gval - gold
            G_index = mAA < mMax ? k : mMax
            G[:, G_index] = Δg
            mAA = mAA + 1
        #    if norm(Δf) < 1e-8 || norm(Δg) < 1e-8
        #        println("ayo, converged")
        #        break
        #    end
        end
        copyto!(fold, fval)
        copyto!(gold, gval)
        if mAA == 0
            println("First iteration")
            copyto!(x, gval)
            println()
        else
            if mAA > 1
                if mAA > mMax
                    qrdelete!(Q, R)
                    mAA = mAA - 1
                end
                for i = 1:mAA-1
                    @show i, mAA
                    R[i, mAA] = dot(Q[:, i], Δf)
                    Δf = Δf - R[i, mAA]*Q[:, i]
                end
            end

            R[mAA, mAA] = norm(Δf, 2)
            @show R, mAA
            @show Δf
            @show cond(R[1:mAA, 1:mAA])
            @show k
            @show mAA
            Q[:, mAA] = Δf/R[mAA, mAA]
            @show Q
            γ = R[1:mAA,1:mAA]\(Q[:, 1:mAA]'*Δf)
            @show - G[:, 1:mAA]*γ
            @show x
            @show gval
            @show G[:, 1:mAA]
            @show γ
            x = gval - G[:, 1:mAA]*γ
            @show x
        end
        println()
    end
    x
end

end # module
