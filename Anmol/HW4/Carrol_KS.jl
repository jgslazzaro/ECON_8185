#Author: João Lazzaro
#This codes solves the agent problem for the Krussell-Smith case
#IT ONLY WORKS FOR EXOGENOUS LABOR OR ENDOGENOUS WITH LOG UTILITY
using Optim, Interpolations, ProgressMeter, LineSearches
using Distributions, Random, DataFrames, GLM, NLsolve
#using Distributed, SharedArrays
include("CRRA_utility.jl")

#Wages functions
R(K::Float64,H::Float64,z::Float64;α=α::Float64,δ=δ::Float64)= z*α*K^(α-1.0)*H^(1.0-α) + (1.0-δ)
w(K::Float64,H::Float64,z::Float64;α=α::Float64) = z*(1.0-α)*K^(α)*H^(-α)

#Law of motion functions for aggregate states
K1(K::Float64,z::Float64;b=b::Array{Float64,2},Z= Z::Array{Float64,1}) = exp(b[findfirst(Z.==z),1]+b[findfirst(Z.==z),2]*log(K))
H0(K::Float64,z::Float64;d=d::Array{Float64,2},Z = Z::Array{Float64,1}) = exp(d[findfirst(Z.==z),1]+d[findfirst(Z.==z),2]*log(K))

#Defining consumption function:
c(a,e, n ,a1,k,h,z) =  R(k,h,z)*a+e*w(k,h,z)*n-a1

#defining optimal labor choice for Log utility case. This comes from labor FOC:
nstar(a,a1,k,h,z;η = η,lbar = lbar) = min(max((1-η)*(a1-R(k,h,z)*a)/w(k,h,z) + lbar*η,0.0),lbar)

#@time nstar(2.0,2.0,K[1],H[1],Z[1])


function ENDOGENOUSGRID_KS(A::Array{Float64,1},A1::Array{Float64,1},E::Array{Float64,1},Z::Array{Float64,1},transmat::Array{Float64,2},states::NTuple,
    K::Array{Float64,1}, H::Array{Float64,1} ,b::Array{Float64,2},d::Array{Float64,2};α=α::Float64,β = β::Float64, η=η::Float64, μ=μ::Float64,
     tol = 1e-6, lbar=lbar::Float64 ,  policy= zeros(nA,nE,nK,nH,nZ,2)::Array{Float64,6},update_policy=0.5::Float64)
    #This function solves the agent problem using the endogenous grid method.
    #A: Individual Asset grid in t!
    #A: Individual Asset grid in t+1
    #E: Individual productivity grid
    #Z: Aggregate shocks grid
    #transmat: transmat object with all the transition matrices
    #states: A tuple which each element is a pair of possible states
    #K: Aggregate capital grid
    #H: Aggregate labor grid
    #b: capital law of motion coefficients
    #d: labor law of motion coefficients

    #OPTIONAL ARGUMENTS
    #update_policy: Damping parameter
    #policy: Initial grid guess for the policy function
    #the othere parameters are self explanatory.
    nA::Int64 = length(A)
    nZ::Int64 = length(Z)
    nE::Int64 = length(E)
    nH::Int64 = length(H)

    #RETURN the grid for the agents policy functions.

    #Guess for policy functions
    itpn = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2],
    extrapolation_bc=Line())
    itpa = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,1],
    extrapolation_bc=Line())
    policy_a(a,e,k,h,z) = itpa(a,e,k,h,z)
    policy_n(a,e,k,h,z;μ=μ) = (e>0)*((μ==1.0)*nstar(a,policy_a(a,e,k,h,z),k,h,z) + (μ!=1.0)*itpn(a,e,k,h,z))


    #Loop st uff:
    policy1= copy(policy) #To store updated values
    prog = ProgressUnknown("Iterations:") #timing and iteration counter, cool stuff
    iteration ::Int64 = 0
    distance::Float64 = 1.0
    dist1 = policy1.-policy #

    while distance > tol
        #the function below returns the new policygrid
    @inbounds @fastmath     innerforloop!(policy1,policy_a,policy_n,b,d;
            A=A,E=E,Z=Z,K=K,H = H,lbar = lbar,A1=A1)

        #check convergence
        if  μ!=1.0 && η!=1.0
            distance = maximum(abs.(policy1-policy))
        else
            distance = maximum(abs.(policy1[:,:,:,:,:,1]-policy[:,:,:,:,:,1]))
        end

        #error if no convergence
        if distance == NaN || distance == Inf
            error("Agent Problem did not converge")
        end

        #update policy function with a damping parameter:
        policy = update_policy .* policy1 .+ (1-update_policy) .* policy

        #update the policy functions:
        itpn = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2],
        extrapolation_bc=Line())
        itpa = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,1],
        extrapolation_bc=Line())
        ProgressMeter.next!(prog; showvalues = [(:Distance, distance)])
        iteration +=1
        if iteration >1000
            break
        end

    end
    ProgressMeter.finish!(prog)
    println("Agent problem finished with a distance of $(distance)")
    return  policy
end




function innerforloop!(policy1::Array{Float64,6},policy_a::Function,policy_n::Function,b,d;
    A1=A1,E=E,Z=Z,K=K,H = H,lbar = lbar,A=A)
    #Returns the updated policy grid given the policies functions
    Threads.@threads for ki = 1:length(K)
        k=K[ki]
        for (zi,z) = enumerate(Z),(hi,h) = enumerate(H),(ei,e) = enumerate(E)
            for (ai,a1) = enumerate(A1) #a1 is assets tommorow
            #Find the level of assets today that generates a1 given the policy functions
                policy1[ai,ei,ki,hi,zi,1],policy1[ai,ei,ki,hi,zi,2] = EE(a1;e=e,z=z,K=k,H=h,b=b,d=d,η=η,Z = Z,E = E,policy_a = policy_a,policy_n = policy_n)
            end

            #sort the asset today (needed for the Interpolation function)
            ordem = sortperm(policy1[:,ei,ki,hi,zi,1])
            #interpolate asset tomorrow as a function of today's:
            itp = LinearInterpolation(policy1[ordem,ei,ki,hi,zi,1],A1[ordem],
            extrapolation_bc=Line())

            #Update the grid:
            for ai = 1:length(A)
                if itp(A[ai])>=A1[1]
                    policy1[ai,ei,ki,hi,zi,1] = itp(A[ai])
                else
                    policy1[ai,ei,ki,hi,zi,1] = 0.0
                end
            end


        end
    end
    return policy1
end

function EE(a1;e=E[e]::Float64,
    z=Z[z]::Float64,K=K[k]::Float64,H=H[h]::Float64,states=states::NTuple{4,Array{Float64,1}},
    b=b::Array{Float64,2},d=d::Array{Float64,2},η=η::Float64,Z = Z,E = E,
    policy_a = policy_a::Function,policy_n = policy_n::Function)
    #a1 is the asset level tommorow
    #Finds assets today as a function of assets tomorrow using the Euler equations
    i::Int64 = findfirstn(states,[z,e]) #find the current state index
    k1::Float64 = K1(K,states[i][1];b=b) #Aggregate states tommorow given today
    h1::Float64 = H0(k1,states[i][1];d=d)
    RHS1 = 0.0 #Find the RHS of the consumption FOC uct'= βE[R uct1 ']
    for e1=1:nE, z1 = 1:nZ #for all possible states tommorow
        j::Int64 = findfirstn(states,[Z[z1],E[e1]]) #find the tommorow state index
        a2 = policy_a(a1,E[e1],k1,h1,Z[z1]) #find assets in t+2 given policy function
        n1 = policy_n(a1,E[e1],k1,h1,Z[z1]) #find labor in t+1 given policy function
        c1 = c(a1,E[e1],n1,a2,k1,h1,Z[z1]) #find consumption in t+1 given policy function
        l1 = lbar - n1 #leisure
        RHS1 += β * transmat[i,j]*R(k1,h1,Z[z1])*uc(c1,l1) #The RHS for the state j given i
    end
    RHS1 = (RHS1/η)^(-1/μ)
    #Find the level of assets today that generates a1 given the policy functions
    if η != 1.0
        a = (RHS1 + a1 - e*w(K,H,z)*(η - (a1*(η - 1))/w(K,H,z)))/(R(K,H,z) + R(K,H,z)*e*(η - 1))
        n = nstar(a,a1,K,H,z,η = η)
    else
        a = (RHS1 + a1 - e*w(K,H,z)*lbar)/R(K,H,z)
        n = e * lbar
    end
    return a,n
end


function KrusselSmithENDOGENOUS(A::Array{Float64,1},A1::Array{Float64,1},
    E::Array{Float64,1},Z::Array{Float64,1},tmat::TransitionMatrix,states::NTuple{4,Array{Float64,1}},
    K::Array{Float64,1},H::Array{Float64,1},  b::Array{Float64,2},d::Array{Float64,2};
    α = α::Float64,β = β::Float64, η = η::Float64, μ=μ::Float64, tol= 1e-6::Float64,
    update_policy=0.5::Float64,updateb= 0.3::Float64,N::Int64=5000,T::Int64=11000,
    discard::Int64=1000,seed::Int64= 2803,lbar=lbar::Float64)
    #This performs KS algorithm
    #A: Individual Asset grid in t!
    #A1: Individual Asset grid in t+1
    #E: Individual productivity grid
    #Z: Aggregate shocks grid
    #tmat: transmat object with all the transition matrices
    #states: A tuple which each element is a pair of possible states
    #K: Aggregate capital grid
    #H: Aggregate labor grid
    #b: capital law of motion coefficients
    #d: labor law of motion coefficients

    #OPTIONAL ARGUMENTS
    #update_policy: Damping parameter for the agent Problem
    #update_b: Damping parameter for the law of motion updates
    #N: Number of agents in the simulation
    #T: Length of the simulated time series
    #discard: number of time periods discarded for ergodicity
    #seed: set the random seed for comparable results
    #policy: Initial grid guess for the policy function
    #the othere parameters are self explanatory.


    #RETURN
    #b: Updated parameter for aggregate capital law of motions
    #d: Updated parameter for aggregate labor law of motions
    #nsim: NxT matrix with simulated labor path for each agent n
    #asim: NxT matrix with simulated assets path for each agent n
    #Ksim: T vector with simulated aggregate  capital
    #Hsim: T vector with simulated aggregate  Labor ,
    #policygrid: Grid with agents policy functions
    #K: new updated grid for aggregate capita (not used for now),
    #R2b,R2d: R-squared of b and d regressions
    #zsimd: T vector with simulated aggregate shocks
    #esim: NxT matrix with idyosincratic employment shock for each agent

    #Getting lengths
    nA::Int64 = length(A)
    nZ::Int64 = length(Z)
    nE::Int64 = length(E)
    nH::Int64 = length(H)
    nK::Int64 = length(K)

    println("Starting Krusell Smith. We are using $(nA) gridpoints for assets and")
    println("a sample of N=$(N), T=$(T). Go somewhere else, this will take a while.")

    transmat::Array{Float64,2} = tmat.P #Getting the transition matrix for the agent

    b=b::Array{Float64,2}
    d=d::Array{Float64,2}
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
    esim = idioshocks(zsim,tmat)
    zsimd::Array{Float64,1} = zsim[discard+1:end] #Discarded simulated values for z

    #predefining variables
    asim::Array{Float64,2} = rand(A[1]:11.6*2,N,T) #the initial assets is between 0 and 2*the mean found by KS
    Ksim::Array{Float64,1} = ones(T)
    Hsim::Array{Float64,1} = ones(T)
    nsim::Array{Float64,2} = ones(N,T)
    R2d::Array{Float64,1} = ones(2)
    R2b::Array{Float64,1} = ones(2)

    #First guessess for Policy
    policygrid::Array{Float64,6} =  ones(nA,nE,nK,nH,nZ,2)
    for (zi,z) = enumerate(Z),(hi,h) = enumerate(H),(ki,k) = enumerate(K),(ei,e)=enumerate(E),(ai,a1)=enumerate(A)
        policygrid[ai,ei,ki,hi,zi,:] = [0.9*a1,e*lbar]
    end
    itpn::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
    extrapolation_bc=Line())
    itpa::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
    extrapolation_bc=Line())
    multiple100::Int64 = 0

    policy_a(a,e,k,h,z) = itpa(a,e,k,h,z)
    policy_n(a,e,k,h,z;μ=μ) = (e>0)*((μ==1.0)*nstar(a,policy_a(a,e,k,h,z),k,h,z) + (μ!=1.0)*itpn(a,e,k,h,z))

    #loop stuff
    dist::Float64 = 1.0
    iteration::Int64 = 0
    b1::Array{Float64,2} = copy(b)
    d1::Array{Float64,2} = copy(d) #to store updated values for b and d

    Ht::Float64 = 1.0

    while (dist>tol)

        println("Solving the agent problem")
        #Solve the agent problem:
        policygrid = ENDOGENOUSGRID_KS(A,A1,E,Z,transmat,states,K, H,b,d;policy= policygrid,update_policy=update_policy,tol = tol)
        itpn = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,2],
        extrapolation_bc=Line())
        itpa = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
        extrapolation_bc=Line())
        println("Agent Problem solved!")


        loading = Progress(T, 1,"Simulating the economy.", 30)   #For loop loading bar minimum update interval: 1 second
        #Simulating the economy
        for t=1:T
            Ksim[t] = mean(asim[:,t]) #Aggregate capital is the mean of the capital decided yesterday
            Ht = H0(Ksim[t],zsim[t];d=d) #Agents believe aggregate labor follows this rule
    Threads.@threads for n=1:N
                if t<=T-1
                    asim[n,t+1] = policy_a(asim[n,t],esim[n,t],Ksim[t],Ht,zsim[t]) #Store each agent asset decision
                end
                    nsim[n,t] = policy_n(asim[n,t],esim[n,t],Ksim[t],Ht,zsim[t]) #Store each agent labor decision
            end

            #Find aggregate labor that clears the market:
            Hsim[t] = mean(nsim[:,t])

            next!(loading) #loading bar stuff
        end

        println("Economy simulated, let's run the regression")
        #Running the regressions
        for i=1:nZ #for each state
            datad = DataFrame(Xd = log.(Ksim[discard+1:end][zsimd.==Z[i]]),
            Yd = log.(Hsim[discard+1:end][zsimd.==Z[i]])) #take log of capital and labor and dataframe it, note that I discard observations
            olsd = lm(@formula(Yd ~ Xd), datad) #regress
            d1[i,:] = coef(olsd) #get coefficients
            R2d[i] = r2(olsd) #and R2

            datab = DataFrame(Xb = log.(Ksim[discard+1:end-1][zsimd[1:end-1].==Z[i]]),
            Yb = log.(Ksim[discard+2:end][zsimd[1:end-1].==Z[i]]))#take log of capital and capital tomorrow and dataframe it, note that I discard observations
            olsb = lm(@formula(Yb ~ Xb), datab) #regress
            b1[i,:] = coef(olsb)#get coefficients
            R2b[i] = r2(olsb) #and R2

        end

        #kmin::Float64 =  max(mean(Ksim)-20.0,eps())
        #kmax::Float64 =  mean(Ksim)+20.0
        #K::Array{Float64,1} = range(kmin,stop = kmax, length = nK).^1

        #check convergence
        dist = maximum(vcat(abs.(b.-b1),abs.(d.-d1)))

        #update law of motions with a damping parameter
        b = updateb.*b1 .+ (1-updateb).*b
        d = updateb.*d1 .+ (1-updateb).*d

        iteration += 1
        println("In iteration $(iteration), law distance is $(dist), Standard Error is $(minimum(sdev))")
        println("b = $(b) and")
        println("d = $(d)")
        println("Aggregate labor mean is $(mean(Hsim))")
        println("Aggregate Capital mean is $(mean(Ksim))")
    end
    #Drop first discard observations:
    Hsim = Hsim[discard+1:end]
    Ksim = Ksim[discard+1:end]
    nsim = nsim[:,discard+1:end]
    asim = asim[:,discard+1:end]
    esim = asim[:,discard+1:end]
    println("Krussell Smith done!")
    return b, d,  nsim, asim, Ksim, Hsim,policygrid,K,R2b,R2d,zsimd,esim
end
