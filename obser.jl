module PhysM

using ..hubbard: lattice, index2xyz, qmcparams
using LinearAlgebra

mutable struct obserStore
	kinetic::Array{Float64}
	doubleoccu::Array{Float64}
	structfactor::Array{Float64}
	function obserStore(Nmeasure::Int)
		k = zeros(Float64,Nmeasure)
		d = zeros(Float64,Nmeasure)
		s = zeros(Float64,Nmeasure)
		new(k,d,s)
	end
end

function kinetic(gttupc::Matrix{Float64},gttdnc::Matrix{Float64}, qmc::qmcparams)
	k = 0.0
	for i=1:qmc.MatDim
		for j = qmc.nnlist[i,:]
			k += gttupc[i,j] + gttupc[j,i] + gttdnc[i,j] + gttdnc[j,i]
		end
	end
	-k / (2 * qmc.MatDim)
end

function structfactor(gttupc::Matrix{Float64}, gttdnc::Matrix{Float64},
					  gttup::Matrix{Float64}, gttdn::Matrix{Float64},qmc::qmcparams)
	sf = 0.0
	q = [pi, pi, pi]
	for i=1:qmc.MatDim
		for j=1:qmc.MatDim
			lxyz = index2xyz(i,qmc.lattice.dim, qmc.lattice.Lxyz) .-
					index2xyz(j, qmc.lattice.dim, qmc.lattice.Lxyz)
			c = real(exp(im*sum(q.*lxyz)))
			spincorr = gttupc[i,i] * gttupc[j,j] + gttupc[i,j] * gttup[i,j] +
                gttdnc[i,i] * gttdnc[j,j] + gttdnc[i,j] * gttdn[i,j] -
                gttdnc[i,i] * gttupc[j,j] - gttupc[i,i] * gttdnc[j,j]
			sf += c*spincorr
		end
	end
	sf / qmc.MatDim^2
end

function sampling(Nmeasure, G_up::Matrix{Float64},
				  G_dn::Matrix{Float64}, qmc::qmcparams, obser::obserStore)
	eye = Matrix(LinearAlgebra.I, qmc.MatDim, qmc.MatDim)
	G_upc = eye - transpose(G_up)
	G_dnc = eye - transpose(G_dn)
	obser.kinetic[Nmeasure] = kinetic(G_upc,G_dnc,qmc)
	obser.doubleoccu[Nmeasure] = sum(diag(G_upc).*diag(G_dnc)) / qmc.MatDim
	obser.structfactor[Nmeasure] = structfactor(G_upc, G_dnc, G_up, G_dn, qmc)
end

end
