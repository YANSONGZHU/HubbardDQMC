include("dqmc.jl")

A = zeros(Float64, 12, 5, 10)
@Threads.threads for i in 1:12
    A[i,:,:] = main()
end
println(sum(A)/(12*5*10))
