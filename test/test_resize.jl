@testitem "System resizing" begin
    using LinearAlgebra, IOCapture

    latvecs = lattice_vectors(1, 1, 1.1, 90, 90, 120)
    cryst = Crystal(latvecs, [[0,0,0]])
    sys = System(cryst, (4, 6, 2), [SpinInfo(1, 1.0)], :dipole)
    set_exchange!(sys, 1.0, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, 1.2, Bond(1, 1, [0, 0, 1]))
    set_exchange!(sys, diagm([0.2, 0.3, 0.4]), Bond(1, 1, [1, 0, 1]))

    for idx in Sunny.all_sites(sys)
        row = idx[2]
        dir = [1, 0, 2mod(row, 2) - 1]
        Sunny.setspin!(sys, Sunny.dipolarspin(sys, dir, idx), idx)
    end

    capt = IOCapture.capture() do
        print_dominant_wavevectors(sys)
    end
    @test capt.output ==
    """
    Dominant wavevectors for spin sublattices:

        [0, 0, 0]              50.00% weight
        [0, 1/2, 0]            50.00%
    """

    capt = IOCapture.capture() do
        suggest_magnetic_supercell([[0,1/2,0]], sys.latsize)
    end
    @test capt.output == """
    Suggested magnetic supercell in multiples of lattice vectors:

        A = [1 0 0; 0 2 0; 0 0 1]

    for wavevectors [[0, 1/2, 0]].
    """

    A1 = [1 0 0; 0 2 0; 0 0 1]
    A2 = [1 0 0; 1 2 0; 0 0 1]
    newsys1 = resize_to_supercell(sys, A1)
    newsys2 = resize_to_supercell(sys, A2)

    @test energy(sys) / prod(sys.latsize) ≈ 2.55
    @test energy(newsys1) / prod(newsys1.latsize) ≈ 2.55
    @test energy(newsys2) / prod(newsys2.latsize) ≈ 2.55
end
