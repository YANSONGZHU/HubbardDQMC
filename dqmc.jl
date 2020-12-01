include("hubbard.jl")
include("svd.jl")
include("data.jl")
include("obser.jl")
include("core.jl")


using ..hubbard: lattice, qmcparams, MatType
using Plots
using ..PhysM: obserStore

function main()
    Lxyz = 4
    dim = 3
    t = 1.0
    U = 6.0
    β = 2.5
    μ = 0.0
    lattice3d = lattice(Lxyz,dim,t,U,β,μ)
    Nwarmup = 10
    Nsweep = 10
    Nbins = 5
    Δτ = 0.05
    Nstable = 2
    qmc = qmcparams(lattice3d, Δτ, Nwarmup, Nsweep, Nbins, Nstable)
    obser = obserStore(Nbins, Nsweep)
    Ns = Int(qmc.Nt/Nstable)
    pst_data = Data.persistent(MatType, Ns, qmc.MatDim)
    tmp_data = Data.temporary(MatType, qmc.MatDim)
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
    # plt = plot([], label = "Double Occupy", zeros(0), xlims = (1,Nbins))
    for i = 1:Nbins
        # println("bin=", i)
        for j = 1:Nsweep
        #    print(j, "...")
            CoreM.sweep!(i, j, G_up, G_dn, B_up_l, B_dn_l, tmp_data,
                         pst_data, qmc, obser, true)
        end
        # meanobser = sum(obser.structfactor[i,:]) / Nsweep
        # push!(plt, i, [meanobser])
        # display(plt)
        # println()
    end
    obser
end
