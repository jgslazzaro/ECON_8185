#Author: João Lazzaro
#This codes solves the agent problem for the Krussell-Smith case
#
using Optim, Interpolations, ProgressMeter, LineSearches, Suppressor
using Distributions, Random, DataFrames, GLM, ForwardDiff,NLsolve
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

#defining optimal labor choice for Log utility case.
nstar(a,a1,k,h,z;η = η,lbar = lbar) = min(max((1-η)*(a1-R(k,h,z)*a)/w(k,h,z) + η,0.0),lbar)

#@time nstar(2.0,2.0,K[1],H[1],Z[1])




function EULER_KS(A::Array{Float64,1},E::Array{Float64,1},Z::Array{Float64,1},transmat::Array{Float64,2},states::NTuple,
    K::Array{Float64,1}, H::Array{Float64,1} ,b::Array{Float64,2},d::Array{Float64,2};α=α::Float64,β = β::Float64, η=η::Float64, μ=μ::Float64,
     tol = 1e-6, lbar=lbar::Float64 ,
     policy= zeros(nA,nE,nK,nH,nZ,2)::Array{Float64,6},update_policy=0.5::Float64,inner_optimizer = LBFGS())
    #A: Individual Asset grid
    #E: Individual productivity grid
    #transmatE: transmat of E
    #Z: Aggregate shocks grid
    #transmatZ: transmat of Z
    #K: Aggregate capital grid
    #H: Aggregate labor grid
    #Vinitial: Guess for the Value function
    nA::Int64 = length(A)
    nZ::Int64 = length(Z)
    nE::Int64 = length(E)
    nH::Int64 = length(H)

    #Guess for policy functions
    #=policy_n::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2],
    extrapolation_bc=Line())
    policy_a::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,1],
    extrapolation_bc=Line()) =#

    itpn = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2],
    extrapolation_bc=Line())
    itpa = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,1],
    extrapolation_bc=Line())
    policy_a(a,e,k,h,z) = itpa(a,e,k,h,z)

    policy_n(a,e,k,h,z) = itpn(a,e,k,h,z)

    policy_n(a,e,k,h,z;μ=μ) = (e>0)*((μ==1.0)*nstar(a,policy_a(a,e,k,h,z),k,h,z) + (μ!=1.0)*itpn(a,e,k,h,z))



    #Loop st uff:
    #last dimension indicates if it is the policy for n or a
    policy1= copy(policy)
    #policy1 = zeros(nA,nE,nK,nH,nZ,2) #To store updated values
    prog = ProgressUnknown("Iterations:")
    iteration ::Int64 = 0
    distance::Float64 = 1.0

    dist1 = policy1.-policy

    while distance > tol
    @inbounds @fastmath     innerforloop!(policy1,policy,policy_a,policy_n,b,d;
            A=A,E=E,Z=Z,K=K,H = H,lbar = lbar)

        dist = copy(dist1)
        #check convergence
        if  μ!=1.0 && η!=1.0
            distance = maximum(abs.(policy1-policy))
        else
            distance = maximum(abs.(policy1[:,:,:,:,:,1]-policy[:,:,:,:,:,1]))
        end


        dist1 = policy1.-policy
        if iteration >1
            φ = dist1./dist
            φ[dist1.<tol] .= 0.0
            φ[φ.>1.0] .=0.5
            φ[0.9.<φ.<=1.0] .= 0.9
        end

        if iteration > 4
            policy = (policy1.- φ.*policy)./(1.0.-φ)
        else
                policy = update_policy .* policy1 .+ (1-update_policy) .* policy
        end


        itpn = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2],
        extrapolation_bc=Line())
        itpa = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,1],
        extrapolation_bc=Line())
        if distance == NaN || distance == Inf
        #    error("Agent Problem did not converge")
        end
        ProgressMeter.next!(prog; showvalues = [(:Distance, distance)])
        iteration +=1

    end
    ProgressMeter.finish!(prog)
    println("Agent problem converged with a distance of $(distance)")
    return  policy
end



function EE!(F,X; a=A[a]::Float64,e=E[e]::Float64,
    z=Z[z]::Float64,K=K[k]::Float64,H=H[h]::Float64,states=states::NTuple{4,Array{Float64,1}},
    b=b::Array{Float64,2},d=d::Array{Float64,2},η=η::Float64,Z = Z,E = E,
    policy_a = policy_a::Function,policy_n = policy_n::Function)

    #policy_a = policy_a::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}},
    #policy_n = policy_n::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}})

    if length(X)==2
        a1,n = X
    elseif e > 0.0 && η==1.0
        a1 = X[1]
        n = lbar
    elseif e > 0.0 && η!=1.0 && μ ==1.0
        a1 = X[1]
        n = nstar(a,a1,K,H,Z)
    elseif e <= 0.0
        a1 = X[1]
        n = 0.0
    else
        error("Length of X should be 2")
    end

    c0  = c(a,e,n,a1,K,H,z)
    l = lbar - n
    LHS1 = uc(c0,l)

    i::Int64 = findfirstn(states,[z,e])
    k1::Float64 = K1(K,states[i][1];b=b)
    h1::Float64 = H0(k1,states[i][1];d=d)
    RHS1 = 0.0
        for e1=1:nE, z1 = 1:nZ
            j::Int64 = findfirstn(states,[Z[z1],E[e1]])
            a2 = policy_a(a1,E[e1],k1,h1,Z[z1])
            n1 = policy_n(a1,E[e1],k1,h1,Z[z1])
            c1 = c(a1,E[e1],n1,a2,k1,h1,Z[z1])
            l1 = lbar - n1
            RHS1 += β * transmat[i,j]*R(k1,h1,Z[z1])*uc(c1,l1)
        end
        F[1] = LHS1 - RHS1
    if length(X)==2
        LHS2 = ul(c0,l)
        RHS2 = e * w(K,H,z)  * uc(c0,l)
        F[2] = LHS2 - RHS2
    end
end



function innerforloop!(policy1::Array{Float64,6},policy::Array{Float64,6},policy_a::Function,policy_n::Function,b,d;
    A=A,E=E,Z=Z,K=K,H = H,lbar = lbar)


    Threads.@threads for ai = 1:length(A) #I think parallel works now...
        a = A[ai]
        for (zi,z) = enumerate(Z),(hi,h) = enumerate(H),(ki,k) = enumerate(K)
            EEsolve!(F,X) = EE!(F,X;a=a,e=E[1],z=z,K=k,H=h,b=b,d=d,policy_a = policy_a,policy_n = policy_n)
            EEsolve2!(F,X) = EE!(F,X;a=a,e=E[2],z=z,K=k,H=h,b=b,d=d,η=η,policy_a = policy_a,policy_n = policy_n)

            #Unemployed agent:
            policy1[ai,1,ki,hi,zi,1] = mcpsolve(EEsolve!,[0.0],[A[end]],[policy[ai,1,ki,hi,zi,1]],autodiff=:forward).zero[1]
            policy1[ai,1,ki,hi,zi,2] = 0.0

            if η !=1.0 && μ!=1.0 #Endogenous labor
                policy1[ai,2,ki,hi,zi,:] = mcpsolve(EEsolve2!,[0.0,0.0],[A[end],lbar],[policy[ai,1,ki,hi,zi,1],lbar-0.01],autodiff=:forward).zero[:]
            elseif η !=1.0 && μ==1.0
                policy1[ai,2,ki,hi,zi,1] = mcpsolve(EEsolve!,[0.0],[A[end]],[policy[ai,2,ki,hi,zi,1]],autodiff=:forward).zero[1]
                policy1[ai,2,ki,hi,zi,2] = nstar(a, policy1[ai,2,ki,hi,zi,1],k,h,z)
            else #labor is exogenous and agent is employed
                policy1[ai,2,ki,hi,zi,1] = nlsolve(EEsolve2!,[policy[ai,2,ki,hi,zi,1]];autodiff=:forward,method = :trust_region).zero[1]
                policy1[ai,2,ki,hi,zi,2] = lbar
            end
        end
    end
    return policy1
end




function KrusselSmith(A::Array{Float64,1},
    E::Array{Float64,1},Z::Array{Float64,1},tmat::TransitionMatrix,states::NTuple{4,Array{Float64,1}},
    K::Array{Float64,1},H::Array{Float64,1},  b::Array{Float64,2},d::Array{Float64,2};
    α = α::Float64,β = β::Float64, η = η::Float64, μ=μ::Float64, tol= 1e-6::Float64,
    update_policy=0.5::Float64,updateb= 0.3::Float64,N::Int64=5000,T::Int64=11000,
    discard::Int64=1000,seed::Int64= 2803,lbar=lbar::Float64,inner_optimizer = LBFGS())


    nA::Int64 = length(A)
    nZ::Int64 = length(Z)
    nE::Int64 = length(E)
    nH::Int64 = length(H)
    nK::Int64 = length(K)

    println("Starting Krusell Smith. We are using $(nA) gridpoints for assets and")
    println("a sample of N=$(N), T=$(T). Go somewhere else, this will take a while.")


    dist::Float64 = 1.0
    iteration::Int64 = 0

    transmat::Array{Float64,2} = tmat.P

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
    asim::Array{Float64,2} = rand(A,N,T)
    Ksim::Array{Float64,1} = ones(T)
    Hsim::Array{Float64,1} = ones(T)
    nsim::Array{Float64,2} = ones(N,T)

    R2d::Array{Float64,1} = ones(2)
    R2b::Array{Float64,1} = ones(2)
    #First guessess for Policy
    policygrid::Array{Float64,6} =  ones(nA,nE,nK,nH,nZ,2)

    println("First, we get an initial guess for the policy functions using a tiny grid:")

    guessA = range(A[1], stop=A[end], length=10).^1.0
    guessH = range(H[1], stop=H[end], length=5).^1.0
    guessK = range(K[1], stop=K[end], length=5).^1.0
    policygridinitial = EULER_KS(guessA,
    E,Z,transmat,states,guessK, guessH,b,d;policy= ones(10,nE,5,5,nZ,2))

    println("We have the initial guess! Let's start the serious stuff:")

    itpn::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((guessA,E,guessK,guessH,Z),policygridinitial[:,:,:,:,:,2],
    extrapolation_bc=Line())
    itpa::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((guessA,E,guessK,guessH,Z),policygridinitial[:,:,:,:,:,1],
    extrapolation_bc=Line())
    multiple100::Int64 = 0

    policy_a(a,e,k,h,z) = itpa(a,e,k,h,z)
    policy_n(a,e,k,h,z;μ=μ) = (e>0)*((μ==1.0)*nstar(a,policy_a(a,e,k,h,z),k,h,z) + (μ!=1.0)*itpn(a,e,k,h,z))

    for a=1:nA,e=1:nE,k=1:nK,h=1:nH,z=1:nZ
        policygrid[a,e,k,h,z,:] = [policy_a(A[a],E[e],K[k],H[h],Z[z]),
                                   policy_n(A[a],E[e],K[k],H[h],Z[z])]
    end
    sdev::Array{Float64,2} = dist .- ones(2,4)
    Ht::Float64 = 1.0

    b1::Array{Float64,2} = copy(b)
    d1::Array{Float64,2} = copy(d)

    tolin::Float64 = 5e-5

    while (dist>tol)

        println("Solving the agent problem")
        policygrid = EULER_KS(A,E,Z,transmat,states,K, H,b,d;policy= policygrid,update_policy=update_policy,tol = tol)

        itpn = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,2],
        extrapolation_bc=Line())
        itpa = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
        extrapolation_bc=Line())

        println("Agent Problem solved!")
        loading = Progress(T, 1,"Simulating the economy.", 30)   # minimum update interval: 1 second
        for t=1:T
            Ksim[t] = max(mean(asim[:,t]),eps())
            Ht = H0(Ksim[t],zsim[t];d=d)
    Threads.@threads for n=1:N
                if t<=T-1
                    asim[n,t+1] = policy_a(asim[n,t],esim[n,t],Ksim[t],Ht,zsim[t])
                end
                    nsim[n,t] = policy_n(asim[n,t],esim[n,t],Ksim[t],Ht,zsim[t])
            end

            #Find aggretate labor that clears the market:
            Hsim[t] = mean(nsim[:,t])

            next!(loading)
        end
        
        println("Economy simulated, let's run the regression")


        for i=1:nZ
            datad = DataFrame(Xd = log.(Ksim[discard+1:end][zsimd.==Z[i]]),
            Yd = log.(Hsim[discard+1:end][zsimd.==Z[i]]))
            olsd = lm(@formula(Yd ~ Xd), datad)
            d1[i,:] = coef(olsd)
            R2d[i] = r2(olsd)
            sdev[i,1:2] = stderror(olsd)

            datab = DataFrame(Xb = log.(Ksim[discard+1:end-1][zsimd[1:end-1].==Z[i]]),
            Yb = log.(Ksim[discard+2:end][zsimd[1:end-1].==Z[i]]))
            olsb = lm(@formula(Yb ~ Xb), datab)
            b1[i,:] = coef(olsb)
            R2b[i] = r2(olsb)

            sdev[i,3:4] = stderror(olsd)


        end


    #    kmin::Float64 =  max(mean(Ksim)-10.0,eps())
    #    kmax::Float64 =  mean(Ksim)+10.0
#        K::Array{Float64,1} = range(kmin,stop = kmax, length = nK).^1

        dist = maximum(vcat(abs.(b.-b1),abs.(d.-d1)))
        iteration += 1
        b = updateb.*b1 .+ (1-updateb).*b
        d = updateb.*d1 .+ (1-updateb).*d

        println("In iteration $(iteration), law distance is $(dist), Standard Error is $(minimum(sdev))")
        println("b = $(b) and")
        println("d = $(d)")
        println("Aggregate labor mean is $(mean(Hsim))")
        println("Aggregate Capital mean is $(mean(Ksim))")


    end

    Hsim = Hsim[discard+1:end]
    Ksim = Ksim[discard+1:end]
    nsim = nsim[:,discard+1:end]
    asim = asim[:,discard+1:end]
    esim = asim[:,discard+1:end]
    println("Krussell Smith done!")
    return b, d,  nsim, asim, Ksim, Hsim,policygrid,K,R2b,R2d,zsimd,esim
end
