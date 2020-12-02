module hubbard

import LinearAlgebra.Diagonal

mutable struct lattice
	Lxyz::Int
	dim::Int
	t::Float64
	U::Float64
	β::Float64
	μ::Float64
end

mutable struct qmcparams
	lattice::lattice
	Δτ::Float64
	Nwarmup::Int
	Nsweep::Int
	Nbins::Int
	α::Float64
	expα::Float64
	expmα::Float64
	MatDim::Int
	Nt::Int
	Nstable::Int
	Ns::Int
	auxfield::Matrix{Int}
	nnlist::Matrix{Int}
	T::Matrix{Float64}
	expT::Matrix{Float64}
	expmT::Matrix{Float64}

	function qmcparams(lattice::lattice, Δτ::Float64, Nwarmup::Int, Nsweep::Int,
					   Nbins::Int, Nstable::Int)
		α = acosh(exp(0.5 * Δτ * lattice.U))
		expα = exp(α)
		expmα = exp(-α)
		MatDim = lattice.Lxyz^lattice.dim
		Nt =  Int(lattice.β / Δτ)
		Ns = Int(Nt/Nstable)
		auxfield = initauxfield(MatDim, Nt)
		nnlist = initnnlist(MatDim, lattice.dim, lattice.Lxyz)
		T = initT(nnlist, MatDim)
		expT = exp(Δτ * T)
		expmT = exp(-Δτ * T)
		new(lattice, Δτ, Nwarmup, Nsweep, Nbins, α, expα, expmα, MatDim, Nt,
		Nstable, Ns, auxfield, nnlist, T, expT, expmT)
	end
end

function initT(list::Matrix{Int}, MatDim::Int)
	Tmatrix = zeros(Float64, MatDim, MatDim)
	for i = 1:MatDim
		for j in list[i,:]
			Tmatrix[i,j] = 1
		end
	end
	Tmatrix
end

function initnnlist(MatDim::Int, dim::Int, Lxyz::Int)
	nnlist = zeros(Int, MatDim, dim*2)
	for i = 1:MatDim
		nnlist[i,:] = neighbor(i, dim, Lxyz)
	end
	nnlist
end

# find the nearest neighbor lattice points of giving index
function neighbor(index::Int, dim::Int, Lxyz::Int)
	nnindex = zeros(Int,2*dim)
	xyz = index2xyz(index, dim, Lxyz)
	for i = 1:dim
		nnxyz = copy(xyz)
		nnxyz[i] = (nnxyz[i] + 1) % Lxyz == 0 ? Lxyz : (nnxyz[i] + 1) % Lxyz
		nnindex[2*i-1] = xyz2index(nnxyz, dim, Lxyz)
		nnxyz = copy(xyz)
		nnxyz[i] = (nnxyz[i] - 1) % Lxyz == 0 ? Lxyz : (nnxyz[i] - 1) % Lxyz
		nnindex[2*i] = xyz2index(nnxyz, dim, Lxyz)
	end
	nnindex
end

# convert the 1D index to coord
function index2xyz(index::Int, dim::Int, Lxyz::Int)
	n = index - 1
	xyz = zeros(Int,dim)
	for i = 1:dim
		xyz[i] = n % Lxyz
		n = n ÷ Lxyz
	end
	xyz .+ 1
end

# convert the coord to 1D index
function xyz2index(xyz::Vector{Int}, dim::Int, Lxyz::Int)
	index = 0
	xyz = xyz.-1
	for i = 1:dim
		index += xyz[i]*Lxyz^(i-1)
	end
	index + 1
end

function initauxfield(MatDim::Int, Nt::Int)
	rand([-1,1], MatDim, Nt)
end

function expV(sigma::Int, auxfield::Vector{Int}, expα::Float64, expmα::Float64)
	expVv = map(spin -> spin == 1 * sigma ? expα : expmα, auxfield)
	Diagonal(expVv)
end

function expV!(sigma::Int, auxfield::AbstractVector{Int},
	expV::Vector{Float64}, expα::Float64, expmα::Float64)
	map!(spin -> spin == 1 * sigma ? expα : expmα, expV, auxfield)
end

end
