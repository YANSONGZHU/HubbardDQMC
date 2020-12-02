include("hubbard.jl")
include("svd.jl")
include("data.jl")
include("obser.jl")
include("core.jl")

using Plots

function main()
    Lxyz = 4
    dim = 3
    t = 1.0
    U = 6
    β = 2.5
    μ = 0.0
    lattice3d = hubbard.lattice(Lxyz,dim,t,U,β,μ)
    Nwarmup = 2000
    Nsweep = 20
    Nbins = 50
    Δτ = 0.1
    Nstable = 1
    qmc = hubbard.qmcparams(lattice3d, Δτ, Nwarmup, Nsweep, Nbins, Nstable)
    obser = PhysM.obserStore(Nbins, Nsweep)
    Ns = Int(qmc.Nt/Nstable)
    pst_data = Data.persistent(Float64, Ns, qmc.MatDim)
    tmp_data = Data.temporary(Float64, qmc.MatDim)
    B_up_l, B_dn_l = CoreM.init_B_mat_list!(qmc.auxfield, qmc.expT, tmp_data,
        qmc.MatDim, qmc.Nt, qmc.expα, qmc.expmα)

    SVDM.init_b_udv_store!(pst_data.B_β_τ_up_udv, Ns, qmc.MatDim, Val(:β_τ))
    SVDM.init_b_udv_store!(pst_data.B_β_τ_dn_udv, Ns, qmc.MatDim, Val(:β_τ))

    SVDM.init_b_udv_store!(pst_data.B_τ_0_up_udv, Ns, qmc.MatDim, Val(:τ_0))
    SVDM.init_b_udv_store!(pst_data.B_τ_0_dn_udv, Ns, qmc.MatDim, Val(:τ_0))

    CoreM.fill_b_udv_store!(B_up_l, pst_data.B_β_τ_up_udv, Nstable, Ns,
        tmp_data.mat, tmp_data.udv, Val(:β_τ))
    CoreM.fill_b_udv_store!(B_dn_l, pst_data.B_β_τ_dn_udv, Nstable, Ns,
        tmp_data.mat, tmp_data.udv, Val(:β_τ))

    G_up = CoreM.Gσττ(pst_data.B_τ_0_up_udv[1], pst_data.B_β_τ_up_udv[1], qmc.MatDim)
    G_dn = CoreM.Gσττ(pst_data.B_τ_0_dn_udv[1], pst_data.B_β_τ_dn_udv[1], qmc.MatDim)
    println("initialization done")
    for i = 1:qmc.Nwarmup
        print(i, "...")
        CoreM.sweep!(0, i, G_up, G_dn, B_up_l, B_dn_l, tmp_data,
                     pst_data, qmc, obser, false)
    end

    println()
    plt = plot([], label = "Double Occupy", zeros(0), xlims = (1,Nbins))
    for i = 1:Nbins
        println("bin=", i)
        @time for j = 1:Nsweep
            print(j, "...")
            CoreM.sweep!(i, j, G_up, G_dn, B_up_l, B_dn_l, tmp_data,
                         pst_data, qmc, obser, true)
        end
        meanobser = sum(obser.structfactor[i,:]) / Nsweep
        push!(plt, i, [meanobser])
        display(plt)
        println()
    end
    println(sum(obser.structfactor) / (Nbins * Nsweep))
    # obser.structfactor
end
