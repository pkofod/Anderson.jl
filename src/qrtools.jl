
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
        #temp = sqrt(R[i, i+1]^2 + R[i+1, i+1]^2)
        temp = hypot(R[i, i+1], R[i+1, i+1])
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
