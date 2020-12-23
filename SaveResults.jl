include("dqmc.jl")
using DelimitedFiles

Nwarmup = 1000
Nmeasure = 2000
threadsnum = 4
kresult = zeros(Nmeasure,threadsnum)
dresult = zeros(Nmeasure,threadsnum)
sresult = zeros(Nmeasure,threadsnum)
U = 4.0
for T = 0.4:0.1:1.0
    print("T = "*string(T)*"\n")
    kResultFile = "DATA\\U 04\\E T" * string(T) * ".dat"
    dResultFile = "DATA\\U 04\\D T" * string(T) * ".dat"
    sResultFile = "DATA\\U 04\\Sπ T" * string(T) * ".dat"
    @time @Threads.threads for i = 1:threadsnum
        kresult[:,i], dresult[:,i], sresult[:,i] = main(Nwarmup, Nmeasure, U, T, i)
    end
    open(kResultFile, "w") do io
           writedlm(io, kresult, '\t')
    end
    open(dResultFile, "w") do io
           writedlm(io, dresult, '\t')
    end
    open(sResultFile, "w") do io
           writedlm(io, sresult, '\t')
    end
end
