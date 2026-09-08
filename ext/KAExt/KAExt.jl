module KAExt

using Adapt: Adapt
using KernelAbstractions: KernelAbstractions, @groupsize, @index, @kernel, @localmem, @synchronize, @uniform
using CUDA: CUDA, @cuda, CUBLAS, CUSOLVER, CuArray, CuDynamicSharedArray,
            CuMatrix, CuPtr, CuVector, StridedCuArray, blockDim, blockIdx,
            launch_configuration, threadIdx
using StaticArrays: StaticArrays, SArray, SMatrix, SVector
using Sunny: Sunny, Bond, Crystal, SpinWaveTheory, System, global_position, 
             intensities_bands

include("System/Types.jl")
include("System/TypesSUN.jl")
include("System/TypesSUNFP32.jl")
include("System/System.jl")
include("System/IncomingBondCSR.jl")
include("SpinWaveTheory/SpinWaveTheory.jl")
include("KPM/MultiplyByHamiltonian.jl")
include("KPM/Lanczos.jl")
include("KPM/SpinWaveTheoryKPM.jl")
include("KPM/MultiplyByHamiltonianBatched.jl")
include("KPM/MultiplyByHamiltonianBatchedSUN.jl")
include("KPM/MultiplyByHamiltonianBatchedSUNFP32.jl")
include("KPM/BatchedDotChains.jl")
include("KPM/BatchedProjection.jl")
include("KPM/BatchedBroadcasts.jl")
include("KPM/LanczosBatched.jl")
include("KPM/SpinWaveTheoryKPMBatched.jl")
include("KPM/LanczosBatchedFP32.jl")

end
