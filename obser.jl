module PhysM

using ..hubbard: lattice, index2xyz, qmcparams
using LinearAlgebra

mutable struct obserStore
	kinetic::Matrix{Float64}
	doubleoccu::Matrix{Float64}
	structfactor::Matrix{Float64}
	function obserStore(Nbins::Int, Nsweep::Int)
		k = zeros(Float64,Nbins,Nsweep)
		d = zeros(Float64,Nbins,Nsweep)
		s = zeros(Float64,Nbins,Nsweep)
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
	-qmc.lattice.t * k / (2 * qmc.MatDim)
end

function structfactor(gttupc::Matrix{Float64}, gttdnc::Matrix{Float64},
					  gttup::Matrix{Float64}, gttdn::Matrix{Float64},qmc::qmcparams)
	sf = 0.0
	for i=1:qmc.MatDim
		for j=1:qmc.MatDim
			lxyz = index2xyz(j,qmc.lattice.dim, qmc.lattice.Lxyz) .-
					index2xyz(i, qmc.lattice.dim, qmc.lattice.Lxyz)
			q = [pi, pi, pi]
			c = (-1)^sum(lxyz)*real(exp(im*prod(q.*lxyz)))
			# c = real(exp(im*prod(q.*lxyz)))
			spincorr = gttupc[j,j] * gttupc[i,i] + gttupc[j,i] * gttup[i,j] +
                gttdnc[j,j] * gttdnc[i,i] + gttdnc[j,i] * gttdn[i,j] -
                gttdnc[j,j] * gttupc[i,i] - gttupc[j,j] * gttdnc[i,i]
			sf += c*spincorr
		end
	end
	sf / qmc.MatDim^2
end

function sampling(Nbins::Int, Nsweep::Int, G_up::Matrix{Float64},
				  G_dn::Matrix{Float64}, qmc::qmcparams, obser::obserStore)
	eye = Matrix(LinearAlgebra.I, qmc.MatDim, qmc.MatDim)
	G_upc = eye - transpose(G_up)
	G_dnc = eye - transpose(G_dn)
	# obser.kinetic[Nbins, Nsweep] = kinetic(G_upc,G_dnc,qmc)
	# obser.doubleoccu[Nbins, Nsweep] = sum(diag(G_upc).*diag(G_dnc)) / qmc.MatDim
	obser.structfactor[Nbins, Nsweep] = structfactor(G_upc, G_dnc, G_up, G_dn, qmc)
end

end