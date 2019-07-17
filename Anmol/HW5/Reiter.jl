#Author: João Lazzaro
#This codes solves the agent problem for the Krussell-Smith case
using Interpolations, ProgressMeter
using Distributions, Random, DataFrames, GLM
#using Distributed, SharedArrays
include("CRRA_utility.jl")

#Wages functions
R(z,k,l;α = α,δ=δ)=  α*z*k^(α-1)*l^(1-α) + 1-δ
w(z,k,l;α = α)=  (1-α)*z*k^α*l^(-α)

#Law of motion functions for aggregate states
K1(K::Float64,z::Float64;b=b::Array{Float64,2},Z= Z::Array{Float64,1}) = exp(b[findfirst(Z.==z),1]+b[findfirst(Z.==z),2]*log(K))
H0(K::Float64,z::Float64;d=d::Array{Float64,2},Z = Z::Array{Float64,1}) = exp(d[findfirst(Z.==z),1]+d[findfirst(Z.==z),2]*log(K))

#Defining asset investment function:
#c(a,e, n ,a1,k,h,z) =  R(k,h,z)*a+e*w(k,h,z)*n-a1
a1star(c,a,e,w,R) = R*a-c+(w*e*nstar(c,e,w))

#defining optimal labor choice. This comes from labor FOC:
nstar(c,e,w;η = η,lbar = lbar) = min(max((lbar - (1-η)/η * c/(w*e)),0.0),lbar)



function ENDOGENOUSGRID(R,w,A::Array{Float64,1},A1::Array{Float64,1},E::Array{Float64,1},
    pdfE;α=α::Float64,β = β::Float64, η=η::Float64, μ=μ::Float64,
     tol = 1e-6, lbar=lbar::Float64 ,  policy= ones(Ap_size,E_size)::Array{Float64,2},update_policy=0.5::Float64,updaterule = false)
    #This function solves the agent problem using the endogenous grid method.
    #A: Individual Asset grid in t!
    #A: Individual Asset grid in t+1
    #E: Individual productivity grid
    #Z: Aggregate shocks grid
    #transmat: transmat object with all the transition matrices
    #states: A tuple which each element is a pair of possible states
    #K: Aggregate capital grid
    #H: Aggregate labor grid

    #OPTIONAL ARGUMENTS
    #update_policy: Damping parameter
    #policy: Initial grid guess for the policy function
    #udpdaterule: false for regular update rule, true for extrapolation (see below)
    #the othere parameters are self explanatory.

    #RETURN the grid for the agents policy functions.

    #Guess for policy functions
    #itpn = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2],
    #extrapolation_bc=Line())
    itpc = LinearInterpolation((A,E),policy, extrapolation_bc=Line())
    policy_c(a,e) = itpc(a,e)
    policy_n(a,e) = nstar(policy_c(a,e),e,w)
    policy_a(a,e;w=w,R=R) = a1star(policy_c(a,e),a,e,w,R)


    #Loop st uff:
    policy1= copy(policy) #To store updated values
    prog = ProgressUnknown("Iterations:") #timing and iteration counter, cool stuff
    iteration ::Int64 = 0
    distance::Float64 = 1.0
    dist1 = policy1.-policy #

    while distance > tol
        #the function below returns the new policygrid
    @inbounds   innerforloop!(policy1,policy_c,policy_n,w,R;
            A1=A1,E=E,lbar = lbar,A=A)

        #check convergence
        distance = maximum(abs.(policy1-policy))

        #error if no convergence
        if distance == NaN || distance == Inf
            error("Agent Problem did not converge")
        end

        #see http://www.econ2.jhu.edu/People/CCarroll/SolvingMacroDSOPs.pdf (also
        #in references directory) section 4.2 for an explanation of the parameter φ
        #it is a clever update rule.
        dist = copy(dist1)
        dist1 = policy1.-policy
        if iteration >1
            φ = dist1./dist
            φ[dist1.<tol] .= 0.0
            φ[φ.>1.0] .=0.5
            φ[0.9.<φ.<=1.0] .= 0.9
        end
        if iteration > 4 && updaterule
            policy = (policy1.- φ.*policy)./(1.0.-φ)
        else
            policy = update_policy*policy1 + (1.0-update_policy)*policy1
        end

        #update the policy functions:
        itpc = LinearInterpolation((A,E),policy, extrapolation_bc=Line())
        ProgressMeter.next!(prog; showvalues = [(:Distance, distance)])
        iteration +=1
        if iteration == 500 || iteration > 1200
            update_policy = rand()
        elseif iteration >10000
            break
        end

    end
    ProgressMeter.finish!(prog)
    println("Agent problem finished with a distance of $(distance)")
    return  policy
end




function innerforloop!(policy1::Array{Float64,2}, policy_c::Function,policy_n,w,R;
    A1=A1,E=E,lbar = lbar,A=A)
    #Returns the updated policy grid given the policies functions
    A0 = copy(A1)
         for (ei,e) = enumerate(E)
            for (ai,a1) = enumerate(A1) #a1 is assets tommorow
            #Find the level of assets and consumption today that generates a1 given the policy functions
                policy1[ai,ei],A0[ai] = EE(a1,w,R;e=e,η=η,policy_c = policy_c)
            end

            #sort the asset today (needed for the Interpolation function)
            ordem = sortperm(A0)
            #interpolate consumption today as a function of today's:
            itpc0 = LinearInterpolation(A0[ordem],policy1[ordem,ei],extrapolation_bc=Line())

            #Update the grid:
            for ai = 1:length(A)
                if A0[1]<=A1[ai]
                    policy1[ai,ei] = itpc0(A1[ai])
                else
                    policy1[ai,ei] = η*(R*A0[ai]- A1[1] + max(w*e*lbar,0.0))
                end
            end
        end
    return policy1
end

function EE(a1,w,R;e=E[e]::Float64,
    E=E, η=η::Float64,policy_c=policy_c,pdfE=pdfE)
    #a1 is the asset level tommorow
    #Finds assets today as a function of assets tomorrow using the Euler equations
    i::Int64 = findfirst(E.==e) #find the current state index
    RHS1::Float64 = 0.0 #Find the RHS of the consumption FOC uct'= βE[R uct1 ']
    for e1=1:length(E) #for all possible states tommorow
        c1::Float64 = policy_c(a1,E[e1]) #find consumption in t+1 given policy function
        a2::Float64 = a1star(c1,a1,E[e1],w,R) #find assets in t+2 given policy function
        n1::Float64 = nstar(c1,E[e1],w) #find labor in t+1 given policy function
        l1::Float64 = lbar - n1 #leisure
        RHS1 += β * pdfE[i,e1]*R*uc(c1,l1) #The RHS for the state j given i
    end

    #Find the level of consumption today that generates a1 given the policy functions
    if e > 0.0
        c = (RHS1/η * ((1-η)/(η*e*w))^(-(1-μ)*(1-η)))^(-1/μ)
    else
        c = (RHS1/η *lbar^(-(1-μ)*(1-η)))^(1/(η*(1-μ)-1))
    end
    #Find the consitent asset level for today (endogenous grid)
    a = c+a1-max(e*w*nstar(c,e,w),0)

    return c,a
end


function weight(a1,a,e,policy,grid)

    a1i::Int64 = findfirst(grid.==a1)
    ai::Int64 = findfirst(grid.==a)

    if a1i<length(grid) && a1<= policy(a,e) <= grid[a1i+1]
        return (grid[a1i+1] - policy(a,e))/(grid[a1i+1] - a1)
    elseif a1i>1 && grid[a1i-1]<=policy(a,e)<= a1
        return (policy(a,e)-grid[a1i-1])/(a1-grid[a1i-1])
    else
        return 0.0
    end
end

function λ1(a1,e1,λ,policy;E=E,grid = Ad,pdfE=pdfE::Array{Float64,2})
    lambda::Float64 = 0.0
    e1i::Int64 = findfirst(E.==e1)
    for ei =1:length(E), ai =1:Ad_size
        lambda += pdfE[ei,e1i] * weight(a1,grid[ai],E[ei],policy,grid) * λ[ai,ei]
    end
    return lambda
end


function invariantλ(λ,policy;Ad=Ad,E=E)
dist::Float64 = 1.0
lambda2::Array{Float64,2}=copy(λ)
    while dist > 1e-8
        for ai=1:length(Ad),ei=1:length(E)
            lambda2[ai,ei] = λ1(Ad[ai],E[ei],λ,policy)
        end
        dist = maximum(abs.(λ-lambda2))
        println(dist)
        λ = copy(lambda2)
    end
    return λ
end
