# Frequently used static types
const Vec3 = SVector{3, Float64}
const Vec5 = SVector{5, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}
const Mat5 = SMatrix{5, 5, Float64, 25}
const CMat3 = SMatrix{3, 3, ComplexF64, 9}
const CVec{N} = SVector{N, ComplexF64}
const HermitianC64 = Hermitian{ComplexF64, Matrix{ComplexF64}}
const HermitianC64Device = Hermitian{ComplexF64, AbstractMatrix{ComplexF64}}

# Convenience for Kronecker-δ syntax. Note that boolean (false, true) result
# acts as multiplicative (0, 1).
@inline δ(x, y) = (x==y)

# Calculates norm(a)^2 without allocating
norm2(a::Number) = abs2(a)
function norm2(a)
    acc = 0.0
    for i in eachindex(a)
        acc += norm2(a[i])
    end
    return acc
end

# Calculates norm(a - b)^2 without allocating
diffnorm2(a::Number, b::Number) = abs2(a - b)
function diffnorm2(a, b)
    @assert size(a) == size(b) "Non-matching dimensions"
    acc = 0.0
    for i in eachindex(a)
        acc += diffnorm2(a[i], b[i])
    end
    return acc
end

function is_integer(x; atol)
    return abs(x - round(x)) < atol
end

function all_integer(xs; atol)
    return all(is_integer(x; atol) for x in xs)
end

# Periodic variant of Base.isapprox. When comparing lattice quantities like
# positions or bonds, prefer is_periodic_copy because it works element-wise.
function isapprox_mod1(x::AbstractArray, y::AbstractArray; opts...)
    @assert size(x) == size(y) "Non-matching dimensions"
    Δ = @. mod(x - y + 0.5, 1) - 0.5
    return isapprox(Δ, zero(Δ); opts...)
end

# Calculates norm(a - b)^2 without allocating
function reduce_kernel(f, op, v0::T, A, B, ::Val{LMEM}, result) where {T, LMEM}
    tmp_local = CUDA.@cuStaticSharedMem(T, LMEM)
    global_index = threadIdx().x
    acc = v0

    # Loop sequentially over chunks of input vector
    while global_index <= length(A)
        element = f(A[global_index], B[global_index])
        acc = op(acc, element)
        global_index += blockDim().x
    end

    # Perform parallel reduction
    local_index = threadIdx().x - 1
    @inbounds tmp_local[local_index + 1] = acc
    sync_threads()

    offset = blockDim().x ÷ 2
    while offset > 0
        @inbounds if local_index < offset
            other = tmp_local[local_index + offset + 1]
            mine = tmp_local[local_index + 1]
            tmp_local[local_index + 1] = op(mine, other)
        end
        sync_threads()
        offset = offset ÷ 2
    end

    if local_index == 0
        result[blockIdx().x] = @inbounds tmp_local[1]
    end

    return
end

function mapreduce_gpu(f::Function, op::Function, A, B)
    OT = Float64
    v0 = 0.0

    threads = 256
    out = CuArray{OT}(undef, (1,))
    CUDA.@cuda threads=threads reduce_kernel(f, op, v0, A, B, Val{threads}(), out)
    Array(out)[1]
end

# Calculates norm(a - b)^2 without allocating
function diffnorm2_cuda(A, B)
    @assert size(A) == size(B) "Non-matching dimensions"
    return mapreduce_gpu(diffnorm2, +, A, B)
end

_hermitianpart(a::Number) = real(a)

function _hermitianpart!(A, n)
    k = threadIdx().x
    i::Int = floor((2*n+1 - sqrt((2n+1)*(2n+1) - 8*k))/2)
    j::Int = k - n*i + i*(i-1)/2
    if i == j
        A[j, j] = _hermitianpart(A[j, j])
    else
        A[i, j] = val = (A[i, j] + adjoint(A[j, i])) / 2
        A[j, i] = adjoint(val)
    end
    return nothing
end

#=
function hermitianpart!(A)
    #require_one_based_indexing(A)
    n = size(A, 1)
    @assert size(A, 2) == n
    CUDA.@cuda threads=div(n*(n+1), 2) _hermitianpart!(A, n)
    return A
end
=#

# Project `v` onto space perpendicular to `n`
@inline proj(v, n) = v - n * ((n' * v) / norm2(n))

# Avoid linter false positives per
# https://github.com/julia-vscode/julia-vscode/issues/1497
kron(a...) = Base.kron(a...)

function tracelesspart(A)
    @assert allequal(size(A))
    return A - tr(A) * I / size(A,1)
end

# https://github.com/JuliaLang/julia/issues/44996
function findfirstval(f, a)
    i = findfirst(f, a)
    return isnothing(i) ? nothing : a[i]
end

# Let F be the matrix with 1's on the antidiagonal. Then AF (or FA) is the
# matrix A with columns (or rows) reversed. If (Q, R) = AF is the QR
# decomposition of AF, then (QF, FRF) is the QL decomposition of A.
function ql_slow(A)
    AF = reduce(hcat, reverse(eachcol(A)))
    Q, R = qr(AF)
    # TODO: Perform these reversals in-place
    QF = reduce(hcat, reverse(eachcol(collect(Q))))
    RF = reduce(hcat, reverse(eachcol(collect(R))))
    FRF = reduce(vcat, reverse(transpose.(eachrow(collect(RF)))))
    return QF, FRF
end
