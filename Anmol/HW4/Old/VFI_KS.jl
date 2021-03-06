#Author: João Lazzaro
#This codes solves the agent problem for the Krussell-Smith case
#
using Optim, Interpolations, ProgressMeter, LineSearches, Suppressor
using Distributions, Random, DataFrames, GLM, ForwardDiff

include("CRRA_utility.jl")

#Wages functions
R(K::Float64,H::Float64,z::Float64;α=α::Float64,δ=δ::Float64)= z*α*K^(α-1.0)*H^(1.0-α) + (1.0-δ)
w(K::Float64,H::Float64,z::Float64;α=α::Float64) = z*(1.0-α)*K^(α)*H^(-α)

#Law of motion functions for aggregate states
K1(K::Float64,z::Float64;b=b::Array{Float64,2},Z= Z::Array{Float64,1}) = exp(b[findfirst(Z.==z),1]+b[findfirst(Z.==z),2]*log(K))
H0(K::Float64,z::Float64;d=d::Array{Float64,2},Z = Z::Array{Float64,1}) = exp(d[findfirst(Z.==z),1]+d[findfirst(Z.==z),2]*log(K))

#Defining consumption function:
c(a::Float64,e::Float64, n ,a1,k::Float64,h::Float64,z::Float64) =  R(k,h,z)*a+e*w(k,h,z)*n-a1
#defining optimal labor choice for Log utility case.
nstar(a,a1,k,h,z;η = η,lbar = lbar) = min(max((1-η)*(a1-R(k,h,z)*a)/w(k,h,z) + η,0.0),lbar)


function VFI_KS(A::Array{Float64,1},E::Array{Float64,1},Z::Array{Float64,1},transmat::Array{Float64,2},states::NTuple,
    K::Array{Float64,1}, H::Array{Float64,1} ,b::Array{Float64,2},d::Array{Float64,2};α=α::Float64,β = β::Float64, η=η::Float64, μ=μ::Float64,
     tol = 1e-6,  inner_optimizer=BFGS(),lbar=lbar::Float64 ,Vgrid::Array{Float64,5} = zeros(nA,nE,nK,nH,nZ),solver = "burro"::String,
     policy= ones(nA,nE,nK,nH,nZ,2)::Array{Float64,6},gridsearchAsize = nA::Int64,gridsearchNsize = 5::Int64,updateV=0.7)
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

    #predefining variables and loop stuff
     #last dimension indicates if it is the policy for n or a
    distance::Float64 = 1.0
    Vgrid1::Array{Float64,5} = copy(Vgrid) #To store V values
    #Guess for Value function
    itpV::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} =
    LinearInterpolation((A,E,K,H,Z),Vgrid,extrapolation_bc=Line()) #linear interpolation of the Value function
    iteration::Int64 = 1
    V(a,e,k,h,z) = itpV(a,e,k,h,z)

    #solver stuff:
    initial::Array{Float64,1} = [A[end]/2, lbar/2]
    lower::Array{Float64,1} = [A[1], 0.0]
    upper::Array{Float64,1} = [Inf,lbar]
    prog = ProgressUnknown("Iterations:")
    while distance > tol
         #Threads.@threads not working since BFGS() is not threadsafe apparently
    @inbounds @fastmath     innerforloop!(Vgrid1,policy,b,d,inner_optimizer,V;
             A=A,E=E,Z=Z,K=K,H = H,lbar = lbar,nA=nA,nZ=nZ,nH=nH,nK=nK)

        distance = maximum(abs.(Vgrid1 .- Vgrid))

        #check convergence
        if iteration == 1000
            updateV = min(updateV - rand(),0.5)
        end


        Vgrid = updateV*Vgrid1+(1-updateV)*Vgrid



        itpV = LinearInterpolation((A,E,K,H,Z),Vgrid,extrapolation_bc=Line())
        if distance == NaN || distance == Inf
            error("VFI did not converge")
        elseif iteration > 5000
            break
        end
        ProgressMeter.next!(prog; showvalues = [(:Distance, distance)])
        iteration +=1

    end
    ProgressMeter.finish!(prog)
    println("VFI finished with a distance of $(distance)")
    return  policy,Vgrid
end
#Expected value function given states and a value function V
function EV(a,e::Float64,z::Float64,k::Float64,
    V,
    b::Array{Float64,2},d::Array{Float64,2}; E=E::Array{Float64,1},
    Z=Z::Array{Float64,1},transmat::Array{Float64,2}=transmat,states=states::NTuple{4,Array{Float64,1}},
    nE=nE::Int64, nZ=nZ::Int64)

    i::Int64 = findfirstn(states,[z,e])
    k1::Float64 = K1(k,states[i][1];b=b)
    h1::Float64 = H0(k1,states[i][1];d=d)
    expected = 0.0
    for e1=1:nE, z1 = 1:nZ
        j::Int64 = findfirstn(states,[Z[z1],E[e1]])
        expected += transmat[i,j]*V(a,E[e1],k1,h1,Z[z1])
        #Tommorrow the value is the capital stock following the law of motion and the expected value for the shocks
    end
    return expected
end


#Functions to be minimized by the solver
#Choosing asset and labor
function Vf(x::AbstractArray,V,b::Array{Float64,2},d::Array{Float64,2};β = β::Float64, a= a::Int64,e = e::Int64,z=z::Int64,h=h::Int64,k::Int64=k,
    A=A::Array{Float64,1},
    E=E::Array{Float64,1},
    K=K::Array{Float64,1},
    H=H::Array{Float64,1},
    Z=Z::Array{Float64,1},lbar = lbar::Float64,μ = μ)

    if μ!=1.0
        if x[1]<0.0 || x[2]<0.0 || c(A[a],E[e],x[2],x[1],K[k],H[h],Z[z])<0
            return -1e20*minimum([x[1],x[2],c(A[a],E[e],x[2],x[1],K[k],H[h],Z[z])])
        elseif x[2] > lbar
            return 1e20 * lbar
        else
            return -(u(c(A[a],E[e],x[2],x[1],K[k],H[h],Z[z]),lbar-x[2]) + β * EV(x[1],E[e],Z[z],K[k],V,b,d))
        end
    else
        nst = nstar(A[a],x[1],K[k],H[h],Z[z])
        return -(u(c(A[a],E[e],nst,x[1],K[k],H[h],Z[z]),lbar-nst) + β * EV(x[1],E[e],Z[z],K[k],V,b,d))
    end
end
function Vf!(ret::Array{Float64,1},x::Array{Array{Float64,1},1},V,b::Array{Float64,2},d::Array{Float64,2};β = β::Float64, a= a::Int64,e = e::Int64,z=z::Int64,h=h::Int64,k::Int64=k,
    A=A::Array{Float64,1},
    E=E::Array{Float64,1},
    K=K::Array{Float64,1},
    H=H::Array{Float64,1},
    Z=Z::Array{Float64,1},lbar=lbar::Float64 )

    for i = 1:length(x)
        ret[i] = Vf(x[i],V,b,d;β = β, a= a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z,lbar= lbar)
    end
    return ret
end
function Vf(x::Array{Array{Float64,1},1},V,b::Array{Float64,2},d::Array{Float64,2};β = β::Float64, a= a::Int64,e = e::Int64,z=z::Int64,h=h::Int64,k::Int64=k,
    A=A,
    E=E::Array{Float64,1},
    K=K::Array{Float64,1},
    H=H::Array{Float64,1},
    Z=Z::Array{Float64,1},lbar=lbar::Float64 )
    ret::Array{Float64,1} = ones(length(x))

    for i = 1:length(x)
        ret[i] = Vf(x[i],V,b,d;β = β, a= a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z,lbar= lbar)
    end
    return ret
end

#choosing only asset (either we know the agent is unemployed or exogenous labor)
function Vf(x::Float64,V,b::Array{Float64,2},d::Array{Float64,2};
β = β::Float64, a= a::Int64,e = e::Int64,z=z::Int64,h=h::Int64,k::Int64=k,
    A=A::Array{Float64,1},E=E::Array{Float64,1},K=K::Array{Float64,1},H=H::Array{Float64,1},
    Z=Z::Array{Float64,1},lbar=lbar::Float64)

    ret::Float64 = 0.0
    if E[e]<=0 #Unemployed agent
        ret = -(u(c(A[a],E[e],0.0,x,K[k],H[h],Z[z]),lbar) + β * EV(x,E[e],Z[z],K[k],V,b,d))
    elseif η == 1.0
        ret = -(u(c(A[a],E[e],lbar,x,K[k],H[h],Z[z]),0.0) + β * EV(x,E[e],Z[z],K[k],V,b,d))
    elseif η != 1.0 && μ==1.0
        nst = nstar(A[a],x,K[k],H[h],Z[z])
        ret = -(u(c(A[a],E[e],nst,x,K[k],H[h],Z[z]),lbar - nst) + β * EV(x,E[e],Z[z],K[k],V,b,d))
    else
        ret = optimize(n::Array{Float64,1}->-(u(c(A[a],E[e],n[1],x,K[k],H[h],Z[z]),lbar-n[1]) +
        β * EV(x,E[e],Z[z],K[k],V,b,d)),0.0,lbar).minimum
    end
    return ret
end



function innerforloop!(Vgrid1::Array{Float64,5},policy::Array{Float64,6},b::Array{Float64,2},d::Array{Float64,2},inner_optimizer,V;
    A=A::Array{Float64,1},E=E::Array{Float64,1},Z=Z::Array{Float64,1},K=K::Array{Float64,1},H = H::Array{Float64,1},
    lbar = lbar::Float64,nA=nA::Int64,nZ=nZ::Int64,nH=n::Int64,nK=nK::Int64)

        #initial = [0.1,lbar-0.1]
     Threads.@threads for a = 1:nA #I think it works now...
         for z=1:nZ,h=1:nH, k=1:nK
            maxV = optimize( x::Float64->Vf(x,V,b,d;a = a,e=1,z=z,h=h,k=k,K=K,lbar=lbar),0.0,A[end])
            if isnan(maxV.minimum) == false
                policy[a,1,k,h,z,:] = [maxV.minimizer,0.0]
                Vgrid1[a,1,k,h,z] = -maxV.minimum
            else
                println("Optim fails in A=$(A[a]),z=$(Z[z]),H=$(H[h]),k=$(K[k])")
            end
            if η!=1.0 && μ!=1.0 #labor is endogenous and the agent is not unemployed
                #solver
                initial = policy[a,2,k,h,z,:]
                maxV = optimize(x->Vf(x,V,b,d;a = a,e=2,z=z,h=h,k=k,K=K,lbar=lbar),initial,inner_optimizer;autodiff=:forward)
                if isnan(maxV.minimum) == false
                    policy[a,2,k,h,z,:] = maxV.minimizer
                    Vgrid1[a,2,k,h,z] = -maxV.minimum
                else
                    println("Optim fails in A=$(A[a]),z=$(Z[z]),H=$(H[h]),k=$(K[k])")
                end
            elseif η != 1.0 && μ==1.0
                initial1 = policy[a,2,k,h,z,1]
                #maxV = optimize( x->Vf(x,V,b,d;a = a,e=2,z=z,h=h,k=k,K=K),[0.0],[A[end]],[initial1],Fminbox(inner_optimizer);autodiff=:forward)
                maxV = optimize( x->Vf(x,V,b,d;a = a,e=2,z=z,h=h,k=k,K=K),0.0,A[end])

                #if isnan(maxV.minimum) == false
                    policy[a,2,k,h,z,:] = [maxV.minimizer[1],nstar(A[a],maxV.minimizer[1],K[k],H[h],Z[z])]
                    Vgrid1[a,2,k,h,z] = -maxV.minimum
                #else
                #    println("Optim fails in A=$(A[a]),z=$(Z[z]),H=$(H[h]),k=$(K[k])")
                #end
            else   #labor is exogenous and agent is employed
                maxV = optimize( x::Float64->Vf(x,V,b,d;a = a,e=2,z=z,h=h,k=k,K=K,lbar=lbar),0.0,A[end])
                if isnan(maxV.minimum) == false
                    policy[a,2,k,h,z,:] = [maxV.minimizer,lbar]
                    Vgrid1[a,2,k,h,z] = -maxV.minimum
                else
                    println("Optim fails in A=$(A[a]),z=$(Z[z]),H=$(H[h]),k=$(K[k])")
                end
            end


        end
    end
    return Vgrid1,policy
end


function KrusselSmith(A::Array{Float64,1},
    E::Array{Float64,1},Z::Array{Float64,1},tmat::TransitionMatrix,states::NTuple{4,Array{Float64,1}},
    K::Array{Float64,1},
    H::Array{Float64,1},
    b::Array{Float64,2},d::Array{Float64,2};α = α::Float64,
    β = β::Float64, η = η::Float64, μ=μ::Float64, tol= 1e-6,
    N::Int64=5000,T::Int64=11000,discard::Int64=1000,seed::Int64= 2803,solver="burro"::String,
    inner_optimizer = BFGS(),lbar=lbar::Float64,ug = ug::Float64,ub = ub::Float64,updateb= 0.3::Float64,updateV=0.7 )


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
    policygrid::Array{Float64,6} =  ones(nA,nE,nK,nH,nZ,2)
    R2d::Array{Float64,1} = ones(2)
    R2b::Array{Float64,1} = ones(2)
    #First guessess for V
    Vgrid::Array{Float64,5} = zeros(nA,nE,nK,nH,nZ)
    for a=1:nA ,e=1:nE,k=1:nK,h=1:nH,z=1:nZ
        Vgrid[a,e,k,h,z] = u(c(A[a],E[e],0.0,0.0,K[k],H[h],Z[z]),lbar)
        policygrid[a,e,k,h,z,:] = [0.9*A[a],E[e] * lbar-0.1]
    end

    itpV::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),Vgrid,
    extrapolation_bc=Line()) #linear interpolation of the Value function
    itpn::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,2],
    extrapolation_bc=Line())
    itpa::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
    extrapolation_bc=Line())
    multiple100::Int64 = 0
    policy_a(a,e,k,h,z) = itpa(a,e,k,h,z)
    policy_n(a,e,k,h,z;μ=μ) = (e>0)*((μ==1.0)*nstar(a,policy_a(a,e,k,h,z),k,h,z) + (μ!=1.0)*itpn(a,e,k,h,z))
    V(a,e,k,h,z) = itpV(a,e,k,h,z)

    sdev::Array{Float64,2} = dist .- ones(2,4)
    Ht::Float64 = 1.0

    b1::Array{Float64,2} = copy(b)
    d1::Array{Float64,2} = copy(d)
    tolin = 1e-4
    while (dist>tol)

        println("Solving the agent problem")
        policygrid,Vgrid =  VFI_KS(A,E,Z,transmat,states,K,H,b,d;solver=solver,
        α = α,β = β, η = η, μ =μ,tol = tolin,  inner_optimizer = inner_optimizer,lbar = lbar,Vgrid=Vgrid,updateV=updateV,policy=policygrid)

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
            Hsim[t] =optimize(Hstar::Float64-> (mean(nsim[:,t])-
            (w(Ksim[t],Hstar,zsim[t])/(zsim[t]*(1-α)*Ksim[t]^α))^(1/(-α)))^2,0.0,lbar).minimizer
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


        dist = max(maximum(abs.(b.-b1)),maximum(abs.(d.-d1)))
        iteration += 1
        b = updateb.*b1 .+ (1-updateb).*b
        d = updateb.*d1 .+ (1-updateb).*d

            println("In iteration $(iteration), law distance is $(dist), Standard Error is $(minimum(sdev))")
        println("b = $(b) and")
        println("d = $(d)")
        println("Aggregate labor mean is $(mean(Hsim))")
        println("Aggregate Capital mean is $(mean(Ksim))")
        if iteration<3
            tolin = tolin/10.0

        end


    end

    Hsim = Hsim[discard+1:end]
    Ksim = Ksim[discard+1:end]
    nsim = nsim[:,discard+1:end]
    asim = asim[:,discard+1:end]
    esim = asim[:,discard+1:end]
    println("Krussell Smith done!")
    return b, d,  nsim, asim, Ksim, Hsim,policygrid,Vgrid,K,R2b,R2d,zsimd,esim
end
