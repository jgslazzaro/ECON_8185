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
c(a::Float64,e::Float64, n::Float64 ,a1::Float64,k::Float64,h::Float64,z::Float64) =  R(k,h,z)*a+e*w(k,h,z)*n-a1


function VFI_KS(A::Array{Float64,1},E::Array{Float64,1},Z::Array{Float64,1},pdf::Array{Float64,2},states::NTuple,
    K::Array{Float64,1}, H::Array{Float64,1} ,b::Array{Float64,2},d::Array{Float64,2};α=α::Float64,β = β::Float64, η=η::Float64, μ=μ::Float64,
     tol = 1e-6,  inner_optimizer=BFGS(),lbar=lbar::Float64 ,Vgrid::Array{Float64,5} = zeros(nA,nE,nK,nH,nZ),
     policy= ones(nA,nE,nK,nH,nZ,2)::Array{Float64,6},gridsearchAsize = nA::Int64,gridsearchNsize = 5::Int64)
    #A: Individual Asset grid
    #E: Individual productivity grid
    #pdfE: pdf of E
    #Z: Aggregate shocks grid
    #pdfZ: pdf of Z
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

    Vgridf::Array{Float64,5} = copy(Vgrid)
    #Guess for Value function
    V::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} =
    LinearInterpolation((A,E,K,H,Z),Vgrid,extrapolation_bc=Line()) #linear interpolation of the Value function
    iteration::Int64 = 1
    multiple5::Int64 = 0

    if gridsearchAsize==nA
        gridsearchA = A# range(A[1],stop = A[end], length = gridsearchAsize )
    else
        gridsearchA = range(A[1],stop = A[end], length = gridsearchAsize )
    end

    gridsearchn = range(0.0,stop = lbar, length = gridsearchNsize )
    gridsearch::Array{Array{Float64,1},1} = fill([1.0,1.0],gridsearchNsize*gridsearchAsize)
    i::Int64 = 1
    for n=1:gridsearchNsize, a=1:gridsearchAsize
        gridsearch[i] = [gridsearchA[a],gridsearchn[n]]
        i+=1
    end
    griddedV::Array{Float64,1} = ones(length(gridsearch))
    indmax::Int64 = 1

    #solver stuff:
    initial::Array{Float64,1} = [A[end]/2, lbar/2]
    lower::Array{Float64,1} = [A[1], 0.0]
    upper::Array{Float64,1} = [Inf,lbar]
    prog = ProgressUnknown("Iterations:")
    while distance > tol
        Vgridf = copy(Vgrid) #store previous loop values to compare

         #Threads.@threads not working since BFGS() is not threadsafe apparently
         for a = 1:nA #I think it works now...
            for z=1:nZ,h=1:nH, k=1:nK, e = 1:nE
                if (η!=1.0 && E[e]>0.0) #labor is not exogenous and the agent is not unemployed
                    #Rough gridsearch for initial guess
                #    Vf!(griddedV,gridsearch,V;β = β, a = a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z,lbar=lbar)
                #    vsearch, indmax = findmax(griddedV)
                #    initial = gridsearch[indmax]
                    #solver
                    maxV = optimize(x::Array{Float64,1}->Vf(x,V;β = β, a = a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z,lbar=lbar),initial,inner_optimizer)

                    #if (maxV.minimizer[1]!= NaN && maxV.minimizer[2]!= NaN && maxV.minimum != NaN)
                        Vgrid[a,e,k,h,z] = maxV.minimum
                        policy[a,e,k,h,z,:] = maxV.minimizer
                #    else
                    #    policy[a,e,k,h,z,:] = initial
                    #    Vgrid[a,e,k,h,z] = vsearch
                #    end


                elseif (η==1.0 && E[e]>0.0)   #labor is exogenous and agent is employed
                    maxV = optimize( x::Float64->Vf(x,V;β = β, a = a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z,lbar=lbar),lower[1],A[end])
                    policy[a,e,k,h,z,:] = [maxV.minimizer,lbar]
                    Vgrid[a,e,k,h,z] = maxV.minimum

                else  #agent is unemployed
                   maxV = optimize( x::Float64->Vf(x,V;β = β, a = a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z,lbar=lbar),lower[1],A[end])
                    policy[a,e,k,h,z,:] = [maxV.minimizer,0]
                    Vgrid[a,e,k,h,z] = maxV.minimum

                end



            end
        end

        #check convergence
        distance = maximum(abs.(Vgrid-Vgridf))
        V = LinearInterpolation((A,E,K,H,Z),Vgrid,extrapolation_bc=Line())
        if distance == NaN || distance == Inf
            error("VFI did not converge")
        end
        ProgressMeter.next!(prog; showvalues = [(:Distance, distance)])
        iteration +=1
    end
    ProgressMeter.finish!(prog)
    println("VFI converged with a distance of $(distance)")
    return  policy,Vgrid
end

#Expected value function given states and a value function V
function EV(a::Float64,e::Float64,z::Float64,k::Float64,
    V::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}};
    b=b::Array{Float64,2},d=d::Array{Float64,2}, E=E::Array{Float64,1},
    Z=Z::Array{Float64,1},pdf::Array{Float64,2}=pdf,states=states::NTuple{4,Array{Float64,1}},
    nE=nE::Int64, nZ=nZ::Int64)

    i::Int64 = findfirstn(states,[z,e])
    k1::Float64 = K1(k,states[i][1];b=b)
    h1::Float64 = H0(k1,states[i][1];d=d)
    expected::Float64 = 0.0
    for e1=1:nE, z1 = 1:nZ
        j::Int64 = findfirstn(states,[Z[z1],E[e1]])
        expected += pdf[i,j]*V(a,E[e1],k1,h1,Z[z1])
        #Tommorrow the value is the capital stock following the law of motion and the expected value for the shocks
    end
    return expected
end


#Functions to be minimized by the solver
#Choosing asset and labor
function Vf(x::Array{Float64,1},V::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}};β = β::Float64, a= a::Int64,e = e::Int64,z=z::Int64,h=h::Int64,k::Int64=k,
    A=A::Array{Float64,1},
    E=E::Array{Float64,1},
    K=K::Array{Float64,1},
    H=H::Array{Float64,1},
    Z=Z::Array{Float64,1},lbar = lbar::Float64)

    if x[1]<0.0 || x[2]<0.0 || x[2]>lbar
        return 1e12
    else
        return -(u(c(A[a],E[e],x[2],x[1],K[k],H[h],Z[z]),lbar-x[2]) + β * EV(x[1],E[e],Z[z],K[k],V))
    end
end
function Vf!(ret::Array{Float64,1},x::Array{Array{Float64,1},1},V::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}};β = β::Float64, a= a::Int64,e = e::Int64,z=z::Int64,h=h::Int64,k::Int64=k,
    A=A::Array{Float64,1},
    E=E::Array{Float64,1},
    K=K::Array{Float64,1},
    H=H::Array{Float64,1},
    Z=Z::Array{Float64,1},lbar=lbar::Float64 )

    for i = 1:length(x)
        ret[i] = Vf(x[i],V;β = β, a= a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z,lbar= lbar)
    end
    return ret
end
function Vf(x::Array{Array{Float64,1},1},V::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}};β = β::Float64, a= a::Int64,e = e::Int64,z=z::Int64,h=h::Int64,k::Int64=k,
    A=A,
    E=E::Array{Float64,1},
    K=K::Array{Float64,1},
    H=H::Array{Float64,1},
    Z=Z::Array{Float64,1},lbar=lbar::Float64 )
    ret::Array{Float64,1} = ones(length(x))

    for i = 1:length(x)
        ret[i] = Vf(x[i],V;β = β, a= a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z,lbar= lbar)
    end
    return ret
end

#choosing only asset (either we know the agent is unemployed or exogenous labor)
function Vf(x::Float64,V::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}};β = β::Float64, a= a::Int64,e = e::Int64,z=z::Int64,h=h::Int64,k::Int64=k,
    A=A::Array{Float64,1},E=E::Array{Float64,1},K=K::Array{Float64,1},H=H::Array{Float64,1},
    Z=Z::Array{Float64,1},lbar=lbar::Float64 )

    ret::Float64 = 0.0
    if E[e]<=0
        ret = -(u(c(A[a],E[e],0.0,x,K[k],H[h],Z[z]),lbar) + β * EV(x,E[e],Z[z],K[k],V))
    else
        ret = -(u(c(A[a],E[e],lbar,x,K[k],H[h],Z[z]),0.0) + β * EV(x,E[e],Z[z],K[k],V))
    end
    return ret
end




function KrusselSmith(A::Array{Float64,1},
    E::Array{Float64,1},Z::Array{Float64,1},pdf::Array{Float64,2},states::NTuple{4,Array{Float64,1}},
    K::Array{Float64,1},
    H::Array{Float64,1},
    b::Array{Float64,2},d::Array{Float64,2};α = α::Float64,
    β = β::Float64, η = η::Float64, μ=μ::Float64, tol= 1e-6,
    iterate = "Policy",N::Int64=5000,T::Int64=11000,discard::Int64=1000,seed::Int64= 2803,
    inner_optimizer = BFGS(),lbar=lbar::Float64 )


    nA::Int64 = length(A)
    nZ::Int64 = length(Z)
    nE::Int64 = length(E)
    nH::Int64 = length(H)
    nK::Int64 = length(K)

    println("Starting Krusell Smith. We are using $(nA) gridpoints for assets and")
    println("a sample of N=$(N), T=$(T). Go somewhere else, this will take a while.")


    dist::Float64 = 1.0
    iteration::Int64 = 0




    b=b::Array{Float64,2}
    d=d::Array{Float64,2}
    #getting the shocks
    Random.seed!(seed)
    Ssim::Array{Array{Float64,1},1} = simMC(states,pdf,T,states[1])
    zsim::Array{Float64,1} =ones(T)
    esim::Array{Float64,2} = ones(N,T)
    for n=1:N
        Random.seed!(seed+n)
        Ssim = simMC(states,pdf,T,states[1])
        for t=1:T
            zsim[t] = Ssim[t][1]
            esim[n,t] = Ssim[t][2]
        end
    end

    zsimd::Array{Float64,1} = zsim[discard+1:end] #Discarded simulated values for z


    #predefining variables
    asim::Array{Float64,2} = ones(N,T).*K[end]
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
    end

    policy_n::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,2],
    extrapolation_bc=Line())
    policy_a::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
    extrapolation_bc=Line())
    V::Interpolations.Extrapolation{Float64,5,Interpolations.GriddedInterpolation{Float64,
    5,Float64,Gridded{Linear},NTuple{5,Array{Float64,1}}},Gridded{Linear},Line{Nothing}} = LinearInterpolation((A,E,K,H,Z),Vgrid,
    extrapolation_bc=Line()) #linear interpolation of the Value function
    multiple100::Int64 = 0

    sdev::Array{Float64,2} = dist .- ones(2,4)
    Ht::Float64 = 1.0

    while (dist>1e-5)

        println("Solving the agent problem")
        policygrid,Vgrid =  VFI_KS(A,E,Z,pdf,states,K,H,b,d;
        α = α,β = β, η = η, μ =μ,tol = tol,  inner_optimizer = inner_optimizer,lbar = lbar,Vgrid=Vgrid)

        policy_n = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,2],
        extrapolation_bc=Line())
        policy_a = LinearInterpolation((A,E,K,H,Z),policygrid[:,:,:,:,:,1],
        extrapolation_bc=Line())

        println("Agent Problem solved!")
        loading = Progress(T, 1,"Simulating the economy.", 30)   # minimum update interval: 1 second
        asim = ones(N,T).*K[end]
        for t=1:T
            Ksim[t] = max(mean(asim[:,t]),0.01)
            #Hsim[t] = mean(nsim[:,t])
            Ht = H0(Ksim[t],zsim[t];d=d)
    Threads.@threads for n=1:N
                if t<=T-1
                    #asim[n,t+1] = policy_a(asim[n,t],esim[n,t],Ksim[t],Hsim[t],zsim[t])
                    asim[n,t+1] = policy_a(asim[n,t],esim[n,t],Ksim[t],Ht,zsim[t])
                end
                    #nsim[n,t] = policy_n(asim[n,t],esim[n,t],Ksim[t],Hsim[t],zsim[t])
                    nsim[n,t] = policy_n(asim[n,t],esim[n,t],Ksim[t],Ht,zsim[t])
            end

            #Find aggretate labor that clears the market:
            Hsim[t] =optimize(Hstar::Float64-> (mean(nsim[:,t])-
            (w(Ksim[t],Hstar,zsim[t])/(zsim[t]*(1-α)*Ksim[t]^α))^(1/(-α)))^2,0.0,lbar).minimizer
            next!(loading)
        end

        println("Economy simulated, let's run the regression")
        bold::Array{Float64,2} = copy(b)
        dold::Array{Float64,2} = copy(d)

        for i=1:nZ
            datad = DataFrame(Xd = log.(Ksim[discard+1:end][zsimd.==Z[i]]),
            Yd = log.(Hsim[discard+1:end][zsimd.==Z[i]]))
            olsd = lm(@formula(Yd ~ Xd), datad)
            d[i,:] = coef(olsd)
            R2d[i] = r2(olsd)
            sdev[i,1:2] = stderror(olsd)

            datab = DataFrame(Xb = log.(Ksim[discard+1:end-1][zsimd[1:end-1].==Z[i]]),
            Yb = log.(Ksim[discard+2:end][zsimd[1:end-1].==Z[i]]))
            olsb = lm(@formula(Yb ~ Xb), datab)
            b[i,:] = coef(olsb)
            R2b[i] = r2(olsb)

            sdev[i,3:4] = stderror(olsd)


        end


        kmin::Float64 =  max(mean(Ksim)-10.0,eps())
        kmax::Float64 =  mean(Ksim)+10.0
        K::Array{Float64,1} = range(kmin,stop = kmax, length = nK).^1

        dist = maximum(vcat(abs.(b.-bold),abs.(d.-dold)))
        iteration +=1


        println("In iteration $(iteration), law distance is $(dist), Standard Error is $(minimum(sdev))")
        println("b = $(b) and")
        println("d = $(d)")
        if (iteration >10 && dist>2.0*minimum(sdev))
            break
        end


    end

    Hsim = Hsim[discard+1:end]
    Ksim = Ksim[discard+1:end]
    nsim = nsim[:,discard+1:end]
    asim = asim[:,discard+1:end]
    println("Krussell Smith done!")
    return b, d,  nsim, asim, Ksim, Hsim,policygrid,Vgrid,K,R2b,R2d,zsim,esim
end
