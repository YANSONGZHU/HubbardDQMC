include("hubbard.jl")
include("svd.jl")
include("data.jl")
include("obser.jl")
include("core.jl")

# using Plots

function main(Nwarmup::Int, Nmeasure::Int, U::Float64, Temp::Float64, displayP::Int)
    Lxyz = 4
    dim = 3
    β = 1/Temp
    lattice3d = hubbard.lattice(Lxyz,dim,U,β)
    Δτ = β/max(20, round(β/sqrt(0.1/U)))
    qmc = hubbard.qmcparams(lattice3d, Δτ)
    obser = PhysM.obserStore(Nmeasure)
    pst_data = Data.persistent(Float64, qmc.Nt, qmc.MatDim)
    tmp_data = Data.temporary(Float64, qmc.MatDim)
    B_up_l, B_dn_l = CoreM.init_B_mat_list!(qmc.auxfield, qmc.expT, tmp_data,
        qmc.MatDim, qmc.Nt, qmc.expα, qmc.expmα)

    SVDM.init_b_udv_store!(pst_data.B_β_τ_up_udv, qmc.Nt, qmc.MatDim, Val(:β_τ))
    SVDM.init_b_udv_store!(pst_data.B_β_τ_dn_udv, qmc.Nt, qmc.MatDim, Val(:β_τ))

    SVDM.init_b_udv_store!(pst_data.B_τ_0_up_udv, qmc.Nt, qmc.MatDim, Val(:τ_0))
    SVDM.init_b_udv_store!(pst_data.B_τ_0_dn_udv, qmc.Nt, qmc.MatDim, Val(:τ_0))

    CoreM.fill_b_udv_store!(B_up_l, pst_data.B_β_τ_up_udv, 1, qmc.Nt,
        tmp_data.mat, tmp_data.udv, Val(:β_τ))
    CoreM.fill_b_udv_store!(B_dn_l, pst_data.B_β_τ_dn_udv, 1, qmc.Nt,
        tmp_data.mat, tmp_data.udv, Val(:β_τ))

    G_up = CoreM.Gσττ(pst_data.B_τ_0_up_udv[1], pst_data.B_β_τ_up_udv[1])
    G_dn = CoreM.Gσττ(pst_data.B_τ_0_dn_udv[1], pst_data.B_β_τ_dn_udv[1])
    if displayP == 1
        print("Warming\n")
    end
    for i = 1:Nwarmup
        if displayP == 1 && i%(Nwarmup/100) == 0
            print('.')
        end
        CoreM.sweep!(i, G_up, G_dn, B_up_l, B_dn_l, tmp_data,
                     pst_data, qmc, obser, false)
    end
    if displayP == 1
        print("Measuring\n")
    end
    # plt = plot([], label = "Double Occupy", zeros(0), xlims = (1,Nmeasure))
    for i = 1:Nmeasure
        if displayP == 1 && i%(Nwarmup/100) == 0
            print('.')
        end
        CoreM.sweep!(i, G_up, G_dn, B_up_l, B_dn_l, tmp_data,
                     pst_data, qmc, obser, true)
        # print(obser.doubleoccu)
        # push!(plt, i, [obser.doubleoccu[i]])
        # display(plt)
    end
    if displayP == 1 & i%(Nwarmup/100) == 0
        println()
    end
    # println(sum(obser.doubleoccu) / Nmeasure)
    return obser.kinetic, obser.doubleoccu, obser.structfactor
end
