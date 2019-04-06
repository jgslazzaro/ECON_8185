#Author: João Lazzaro
#This codes solves the agent problem for the Krussell-Smith case
#
using Optim, Interpolations
include("CRRA_utility.jl")
#Uncomment the section below to test this code


function VFI_KS(A::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    E::Array{Float64,1},pdfE::Array{Float64,2},Z::Array{Float64,1},pdfZ::Array{Float64,2},
    K::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    H::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}} ,
    K1::Function,H1::Function,b::Array{Float64,2},d::Array{Float64,2},Vinitial::Function;
    α::Float64 = α,β::Float64 = β, η::Real = η, μ::Real =μ,inner_optimizer = NelderMead(),
     tol = 1e-6, iterate::String = "Value",R::Function= R,w::Function= w,damp::Float64=2/3)
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

    #Constructing the stochastic pdf
    pdf = [πz*πe for πz in pdfZ, πe in pdfE] #nZxnZxnExnE
    pdf = permutedims(pdf, (1,3,2,4)) #nZxnExnZxnE

    #Guess for Value function
    V(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = Vinitial(a,e,K,H,z) #u(c(a,e,0.0,a,k,h,z),1.0)

    #Expected value function given states and a value function V
    function EV(a::Float64,e::Float64,z::Float64,k::Float64;b::Array{Float64,2}=b,d::Array{Float64,2}=d,
        E::Array{Float64,1}=E,
        H::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=H,
        K::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=K,
        Z::Array{Float64,1}=Z,
        pdf::Array{Float64,4}=pdf, nE::Int64 = nE, nZ::Int64=nZ)
        i = findfirst(E.==e)
        j = findfirst(Z.==z)
        expected = 0.0
        for e1=1:nE, z1 = 1:nZ
            expected += pdf[j,i,z1,e1]*V(a,E[e1],K1(k,z1;b=b),H1(k,z1;d=d),Z[z1])
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
    iteration = 1
    multiple5 = 0
    while distance > tol
        Vgridf = copy(Vgrid) #store previous loop values to compare
        policy_old = copy(policy)
        Threads.@threads for a = 1:nA #Parallel for!!
            for z=1:nZ,h=1:nH, k=1:nK, e = 1:nE
                Vsolver(x::Array{Float64,1}) = Vf(x::Array{Float64,1};β = β, a = a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z)
                Vsolver(x::Float64) = Vf(x::Float64;β = β, a = a,e=e,z=z,h=h,k=k,A=A,E=E,K=K,H=H,Z=Z)
                #solver stuff:
                if η != 1 && E[e]>0
                    initial = [1., 0.8]
                    lower = [A[2], 0.0]
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
                    maxV = optimize( Vsolver,lower,upper)
                    policy[a,e,k,h,z,:] = [maxV.minimizer,0]
                else #labor is exogenous and agent is employed
                    maxV = optimize( Vsolver,lower,upper)
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
        V(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = LinearInterpolation((A,E,K,H,Z),Vgrid*(1-damp).+damp*Vgridf, extrapolation_bc=Line())(a,e,k,h,z) #linear interpolation of the Value function
        if div(iteration,10)>multiple5 #Only display a message each 10 iterations
            println("In iteration $(iteration) distance is $(distance)")
            multiple5 = div(iteration,10)
        end
        iteration +=1
    end
    #Finally, find labor, capital and consumption policies interpolations:
    policy_n(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2], extrapolation_bc=Line())(a,e,k,h,z)
    policy_a(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,1], extrapolation_bc=Line())(a,e,k,h,z)
    policy_c(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = c(a,e,policy_n(a,e,k,h,z),policy_a(a,e,k,h,z),k,h,z)
    return  policy_a,policy_n,policy_c,V,policy
end


function KrusselSmith(A::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    E::Array{Float64,1},pdfE::Array{Float64,2},Z::Array{Float64,1},pdfZ::Array{Float64,2},
    K::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    H::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    K1::Function,H1::Function,b::Array{Float64,2},d::Array{Float64,2};α::Float64 = α,
    β::Float64 = β, η::Real = η, μ::Real=μ,inner_optimizer = BFGS(), tol::Float64= 1e-6,
    iterate::String = "Policy",N::Int64=10000,T::Int64=1500,discard::Int64=500,seed::Int64= 2803, R::Function=R,w::Function=w,
    damp::Float64=2/3,damp2::Float64 = 2/3)
    dist = 1.0
    iteration = 0
    nZ = length(Z)

    policy_a = 1
    policy_n = 1
    policy_c = 1
    V = 1
    #getting the shocks
    Random.seed!(seed)
    zsim = simMC(Z,pdfZ,T,Z[1])
    esim=ones(N,T)
    Threads.@threads for n=1:N
        esim[n,:] =  simMC(E,pdfE,T,E[Int64(length(E)/2)])
    end

    #predefining variables
    asim = ones(N,T)
    Ksim =ones(T)
    Hsim =ones(T)
    Ksim[1] = mean(asim[:,1])
    csim = ones(N,T)
    nsim = ones(N,T)
    policygrid =  ones(nA,nE,nK,nH,nZ,2)
    #First guess for V
    #V(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = 0.0
    while dist>tol
        V(a::Float64,e::Float64,k::Float64,h::Float64,z::Float64) = 0.0
        println("Solving the agent problem")
        policy_a,policy_n,policy_c,V,policygrid = VFI_KS(A,E,pdfE,Z,pdfZ,K,H ,K1,H1,b,d,V;
        α = α,β = β, η = η, μ =μ,inner_optimizer = inner_optimizer, tol = tol, iterate = iterate,damp = damp)
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

        for i=1:nZ
        X = hcat(ones(T-discard-1),log.(Ksim[discard+1:end-1]))
        b = X'*X \ X'*log.(Ksim[discard+2:end])

        X = hcat(ones(T-discard-1),log.(Hsim[discard+1:end-1]))
        d = X'*X \ X'*log.(Hsim[discard+2:end])

        dist =1/damp2 * maximum(max(abs.(b-bold),abs.(d-dold)))
        b = (1-damp2)*bold .+ damp2*b
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
