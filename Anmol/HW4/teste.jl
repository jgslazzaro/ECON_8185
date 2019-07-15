
#First guessess for Policy


policygrid= zeros(nA,nE,nK,nH,nZ,2)

#Guess for policy functions
itpn = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,2],
extrapolation_bc=Line())

itpa =LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
extrapolation_bc=Line())
policy_a(a,e,k,h,z) =  itpa(a,e,k,h,z)
policy_n(a,e,k,h,z;μ=μ) = (e>0)*((μ==1.0)*nstar(a,policy_a(a,e,k,h,z),k,h,z) + (μ!=1.0)*itpn(a,e,k,h,z))

include("Carrol_KS.jl")



println("Starting Krusell Smith. We are using $(nA) gridpoints for assets and")
println("a sample of N=$(N), T=$(T). Go somewhere else, this will take a while.")

transmat = tmat.P #Getting the transition matrix for the agent
seed = 1234

d=d
#getting the shocks
Random.seed!(seed)
if rand()>0.5
    Random.seed!(seed)
    zsim = simMC(Z,tmat.Pz,T,Z[1])
else
    Random.seed!(seed)
    zsim = simMC(Z,tmat.Pz,T,Z[2])
end
Random.seed!(seed)
esim  = idioshocks(zsim,tmat)
zsimd = zsim[discard+1:end] #Discarded simulated values for z

#predefining variables
asim = rand(K[1]:0.1:K[end],N,T) #the initial assets will generate aggregate assets in the grid
Ksim = ones(T)
Hsim = ones(T)
nsim = ones(N,T)
R2d = ones(2)
R2b = ones(2)


policygrid = ENDOGENOUSGRID_KS(A,A1,E,Z,transmat,states,K, H,b,d;policy= policygrid,update_policy=0.9,tol = 1e-6,updaterule = true)
itpn = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,2],
extrapolation_bc=Line())
itpa = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
extrapolation_bc=Line())
println("Agent Problem solved!")

maximum(policygrid[:,:,:,:,:,2])

using Plots

plot(A,policygrid[:,2,3,3,2,2])


loading = Progress(T, 1,"Simulating the economy.", 30)   #For loop loading bar minimum update interval: 1 second
#Simulating the economy
t=1
#for t=1:T
Ksim[t] = mean(asim[:,t]) #Aggregate capital is the mean of the capital decided yesterday
    #First guess for aggregate labor:
Ht = H0(Ksim[t],zsim[t];d=d)
    #Find aggregate labor that clears the market:
internaldist = 10.0
its = 0
#    while internaldist>1e-6 && its < 100
Threads.@threads for n=1:N
    nsim[n,t] = policy_n(asim[n,t],esim[n,t],Ksim[t],Ht,zsim[t]) #Store each agent labor decision
end
Hsim[t] = mean(nsim[:,t])
internaldist = abs(Hsim[t] - Ht)
Ht = Hsim[t]
its+=1
end

        if t<T
            asim[:,t+1] .= policy_a.(asim[:,t],esim[:,t],Ksim[t],Ht,zsim[t]) #Store each agent asset decision
        end
    next!(loading) #loading bar stuff
end



include("functions.jl")
seed = 12
#getting the shocks
#Random.seed!(seed)
if rand()>0.5
    Random.seed!(seed)
    zsim = simMC(Z,tmat.Pz,T,Z[1])
else
    Random.seed!(seed)
    zsim = simMC(Z,tmat.Pz,T,Z[2])
end
Random.seed!(seed)

mean(zsim)


@time esim1  = idioshocks(zsim,tmat)
esim2 = idioshocks(zsim,tmat)
esim1 == esim2

#zsimd = zsim[discard+1:end] #Discarded simulated values for z

meanUgood = 1-mean(esim[:,zsim.==Z[2]])
meanUbad = 1-mean(esim[:,zsim.==Z[1]])
