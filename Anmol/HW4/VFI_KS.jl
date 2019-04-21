#Author: João Lazzaro
#This codes solves the agent problem for the Krussell-Smith case
#
using Optim, Interpolations
include("CRRA_utility.jl")
#Uncomment the section below to test this code


function VFI_KS(A::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    E::Array{Float64,1},Z::Array{Float64,1},pdf::Array{Float64,2},states,
    K::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    H::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}} ,
    K1::Function,H1::Function,b::Array{Float64,2},d::Array{Float64,2},Vinitial::Function;
    α::Float64 = α,β::Float64 = β, η::Float64 = η, μ::Float64 =μ,
     tol = 1e-6, iterate::String = "Value",R::Function= R,w::Function= w,damp::Float64=2/3,verbose::Bool=false)
    #A: Individual Asset grid
    #E: Individual productivity grid
    #pdfE: pdf of E
    #Z: Aggregate shocks grid
    #pdfZ: pdf of Z
    #K: Aggregate capital grid
    #H: Aggregate labor grid
    #Vinitial: Guess for the Value function
    nA = length(A)
    nZ = length(Z)
    nE = length(E)
    nH = length(H)
    #Defining consumption function:
    c(a::Float64,e::Float64, n::Float64 ,a1::Float64,k::Float64,h::Float64,z::Float64) =  R(k,h,z)*a+e*w(k,h,z)*n-a1



    #Guess for Value function
    V(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = Vinitial(a,e,K,H,z) #u(c(a,e,0.0,a,k,h,z),1.0)

    #Expected value function given states and a value function V
    function EV(a::Float64,e::Float64,z::Float64,k::Float64;b::Array{Float64,2}=b,d::Array{Float64,2}=d,
        E::Array{Float64,1}=E,
        H::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=H,
        K::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=K,
        Z::Array{Float64,1}=Z,
        pdf::Array{Float64,2}=pdf,states=states, nE::Int64 = nE, nZ::Int64=nZ)
        i = findfirstn(states,[z,e])

        k1 = K1(k,states[i][1];b=b)
        h1 = H1(k,states[i][1];d=d)
        expected = 0.0
        for e1=1:nE, z1 = 1:nZ
            j = findfirstn(states,[Z[z1],E[e1]])
            expected += pdf[i,j]*V(a,E[e1],k1,h1,Z[z1])
            #Tommorrow the value is the capital stock following the law of motion and the expected value for the shocks
        end
        return expected
    end
    #Functions to be minimized by the solver
    #Choosing asset and labor
    function Vf(x::Array{Float64,1};β::Float64 = β, a::Int64= a,e::Int64=e,z::Int64=z,h::Int64=h,k::Int64=k,
        A::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=A,
        E::Array{Float64,1}=E,
        K::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=K,
        H::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=H,Z::Array{Float64,1}=Z)
        return -(u(c(A[a],E[e],x[2],x[1],K[k],H[h],Z[z]),1-x[2]) + β * EV(x[1],E[e],Z[z],K[k]))
    end
    #choosing only asset (either we know the agent is unemployed or exogenous labor)
    function Vf(x::Float64;β::Float64 = β, a::Int64= a,e::Int64=e,z::Int64=z,h::Int64=h,k::Int64=k,
        A::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=A,
        E::Array{Float64,1}=E,
        K::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=K,
        H::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=H,Z::Array{Float64,1}=Z)
        if E[e]<=0
            return -(u(c(A[a],E[e],0.0,x,K[k],H[h],Z[z]),1.0) + β * EV(x,E[e],Z[z],K[k]))
        else
            return -(u(c(A[a],E[e],1.0,x,K[k],H[h],Z[z]),0.0) + β * EV(x,E[e],Z[z],K[k]))
        end
    end

    #predefining variables and loop stuff
    policy = ones(nA,nE,nK,nH,nZ,2) #last dimension indicates if it is the policy for n or a
    distance = 1
    Vgrid = zeros(nA,nE,nK,nH,nZ)
    Vgridf = zeros(nA,nE,nK,nH,nZ)
    itpV = LinearInterpolation((A,E,K,H,Z),Vgrid*(1-damp).+damp*Vgridf, extrapolation_bc=Line())
    iteration = 1
    multiple5 = 0
    inner_optimizer = LBFGS()
    while distance > tol
        Vgridf = copy(Vgrid) #store previous loop values to compare

        Threads.@threads  for a = 1:nA #Parallel for!!
            for z=1:nZ,h=1:nH, k=1:nK, e = 1:nE
                #Vsolver(x::Array{Float64,1}) = Vf(x::Array{Float64,1};β = β, a = a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z)
                #Vsolver(x::Float64) = Vf(x::Float64;β = β, a = a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z)
                #solver stuff:
                if η != 1 && E[e]>0
                    initial = [A[end]/2, 0.8]
                    lower = [A[2], eps(0.)]
                    upper = [A[end-1], 1.0]
                else
                    initial = 1.
                    lower = A[2]
                    upper = A[end-1]
                end


                if η!=1 && E[e]>0 #labor is not exogenous and the agent is not unemployed
                   maxV = optimize( Vsolver, lower,upper,initial,Fminbox(inner_optimizer))
                    policy[a,e,k,h,z,:] = maxV.minimizer
                elseif E[e]<=0 #agent is unemployed
                    #maxV = optimize( Vsolver,lower,upper)
                    maxV = optimize( Vf,lower,upper)
                    policy[a,e,k,h,z,:] = [maxV.minimizer,0]
                else #labor is exogenous and agent is employed
                    #maxV = optimize( Vsolver,lower,upper)
                    maxV = optimize( Vf,lower,upper)
                    policy[a,e,k,h,z,:] = [maxV.minimizer,1]
                end
                Vgrid[a,e,k,h,z] = maxV.minimum
            end
        end
        #check convergence
        if iterate == "Value"
            distance = maximum(1/damp*abs.(Vgrid-Vgridf))
        else
            distance = maximum(abs.(policy-policy_old))
        end

        if distance == NaN
            error("VFI did not converge")
        end
        itpV = LinearInterpolation((A,E,K,H,Z),Vgrid*(1-damp).+damp*Vgridf, extrapolation_bc=Line())
        V(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = itpV(a,e,k,h,z) #linear interpolation of the Value function
        if div(iteration,10)>multiple5 #&& verbose==false)#Only display a message each 10 iterations
            println("In iteration $(iteration) distance is $(distance)")
            multiple5 = div(iteration,10)
        else
        #    println("In iteration $(iteration) distance is $(distance)")
        end
        iteration +=1
    end
    #Finally, find labor, capital and consumption policies interpolations:
    itpn = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2], extrapolation_bc=Line())
    policy_n(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = itpn(a,e,k,h,z)
    itpa = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,1], extrapolation_bc=Line())
    policy_a(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = itpa(a,e,k,h,z)

    policy_c(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = c(a,e,policy_n(a,e,k,h,z),policy_a(a,e,k,h,z),k,h,z)
    return  policy_a,policy_n,policy_c,V,policy
end


function KrusselSmith(A::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    E::Array{Float64,1},Z::Array{Float64,1},pdf::Array{Float64,2},states,
    K::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    H::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    K1::Function,H1::Function,b::Array{Float64,2},d::Array{Float64,2};α::Float64 = α,
    β::Float64 = β, η::Float64 = η, μ::Float64=μ, tol::Float64= 1e-6,
    iterate::String = "Policy",N::Int64=10000,T::Int64=1500,discard::Int64=500,seed::Int64= 2803, R::Function=R,w::Function=w,
    damp::Float64=2/3,damp2::Float64 = 2/3,verbose::Bool = false)

    dist = 1.0
    iteration = 0
    nZ = length(Z)


    policy_a(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64)  = 1.0
    policy_n(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64)  = 1.0
    policy_c(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64)  = 1.0
    V(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64)  = 1.0
    #getting the shocks
    Random.seed!(seed)
    Ssim = simMC(states,pdf,T,states[1])
    zsim =ones(T)
    esim = ones(N,T)
    Threads.@threads for n=1:N
        Ssim = simMC(states,pdf,T,states[1])
        for t=1:T
            zsim[t] = Ssim[t][1]
            esim[n,t] = Ssim[t][2]
        end
    end

    zsimd = zsim[discard+1:end] #Discarded simulated values for z

    #predefining variables
    asim = ones(N,T)
    Ksim =ones(T)
    Hsim =ones(T)
    Ksim[1] = mean(asim[:,1])
    csim = ones(N,T)
    nsim = ones(N,T)
    policygrid =  ones(nA,nE,nK,nH,nZ,2)
    #First guess for V

    while dist > tol

        println("Solving the agent problem")
       policy_a,policy_n,policy_c,V,policygrid =  VFI_KS(A,E,Z,pdf,states,K,H ,K1,H1,b,d,V;
        α = α,β = β, η = η, μ =μ,tol = tol, iterate = iterate,damp = damp, verbose=verbose)
        println("Agent Problem solved!")


        for t=1:T
            Ksim[t] = mean(asim[:,t])
            Threads.@threads for n=1:N
                if t<=T-1
                    asim[n,t+1] = policy_a(asim[n,t],esim[n,t],Ksim[t],Hsim[t],zsim[t])
                end
                    nsim[n,t] = policy_n(asim[n,t],esim[n,t],Ksim[t],Hsim[t],zsim[t])
                    #csim[n,t] = policy_c(asim[n,t],esim[n,t],Ksim[t],Hsim[t],zsim[t])
            end
            Hsim[t] = mean(nsim[:,t])
        end

        bold = copy(b)
        dold = copy(d)
        Xb = hcat(ones(T-discard-1),log.(Ksim[discard+1:end-1]))
        Xd = hcat(ones(T-discard-1),log.(Hsim[discard+1:end-1]))

        for i=1:nZ
            b[i,:] = Xb[zsimd[1:end-1].==Z[i],:]'*Xb[zsimd[1:end-1].==Z[i],:] \
            Xb[zsimd[1:end-1].==Z[i],:]'*log.(Ksim[discard+2:end][zsimd[1:end-1].==Z[i]])

            d[i,:] = Xd[zsimd[1:end-1].==Z[i],:]'*Xd[zsimd[1:end-1].==Z[i],:] \
            Xd[zsimd[1:end-1].==Z[i],:]'*log.(Hsim[discard+2:end][zsimd[1:end-1].==Z[i]])
        end

        dist =1/damp2 * maximum(vcat(abs.(b.-bold),abs.(d.-dold)))
        b = (1-damp2)*bold .+ damp2*b
        d = (1-damp2)*dold .+ damp2*d
        iteration +=1
        println("In iteration $(iteration), law distance is $(dist)")
    end
    csim = csim[:,discard+1:end]
    Hsim = Hsim[discard+1:end]
    Ksim = Ksim[discard+1:end]
    nsim = asim[:,discard+1:end]
    asim = asim[:,discard+1:end]
    println("Krussell Smith done!")
    return policy_a, policy_n, policy_c, V, b, d, csim, nsim, asim, Ksim, Hsim,policygrid
end
