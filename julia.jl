include("dqmc.jl")

Nwarmup = 100
Nsweep = 20
Nbins = 10
threadsnum = 4
result = zeros(threadsnum,Nbins,Nsweep)

@time @Threads.threads for i = 1:threadsnum
    result[i,:,:] = main(Nwarmup, Nsweep, Nbins)
end
println(sum(result)/(threadsnum * Nbins * Nsweep))
