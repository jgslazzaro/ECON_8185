
using Optim, Interpolations, Plots
include("CRRA_utility.jl")

#Uncomment the section below to test this code



using Distributions
function Tauchen(ρ,σ,Y,μ = 0,m = 3)
    #This function is to discretize an AR(1) process following Tauchen(1986) method
    # y_{t+1} = μ + ρy_t + ϵ
    #ϵ~N(0,σ^2)
    #Y is the number of y states
if Y>1
    ybar = μ/(1-ρ)
    ymax= ybar + m*(σ^2/(1-ρ^2))^(1\2) #maximum y
    ymin= ybar - m*(σ^2/(1-ρ^2))^(1\2) #minimum y

    Δ = (ymax-ymin)/(Y-1)# #distance between each y
    y=ymin:Δ:ymax #vector of possible states of p

    d=Normal()

    pdfY=ones(Y,Y) #preallocate memory and create the transition matrix in the following loop
    for i in 1:Y
        pdfY[i,1]=cdf(d,(y[1] + Δ/2 -ρ*y[i]) / σ^0.5);
        pdfY[i,Y]=1-cdf(d,(y[Y] -Δ/2 - ρ*y[i]) / σ^0.5);
        for j in 2:Y-1
            pdfY[i,j]=cdf(d,(y[j] + Δ/2 - ρ*y[i])/σ^0.5) - cdf(d,(y[j] - Δ/2 - ρ*y[i]) / σ^0.5);
        end
    end
else
    y=μ
    pdfY=1
end

    return pdfY, y
end

c(k::Real,z::Real, h::Real ,k1::Real;δ::Float64=δ,θ::Float64=θ) =  k^θ*(z*h)^(1-θ )  -(1+γz) *(1+γn)k1  + (1-δ)*k

function VFI(A,E,pdfE;α = α,β = β, η = η, μ =μ,inner_optimizer = BFGS(), tol = 1e-6)
    #Defining consumption function:
    nA = length(A)
    nE = length(E)


    #Guess for Value function
    V(a,e) = 0
    #Expected value function given states and a value function V
    function EV(a,e;E=E,pdfE = pdfE, nE = nE)
        i = findfirst(E.==e)
        expected = 0.0
        for e1 = 1:nE
            expected += pdfE[i,e1]*V(a,E[e1])
        end
        return expected
    end



    #predefining variables and loop stuff
    policy = ones(nA,nE,2) #last dimension indicates if it is the policy for n or a
    distance = 1
    Vgrid = zeros(nA,nE)
    while distance > tol
        Vgridf = copy(Vgrid)

        #policy_old = copy(policy)

        Threads.@threads  for a = 1:nA #Parallel for!!
            for e = 1:nE
                #solver stuff:
                if η != 1.0 && E[e]>0
                    initial = [A[end]/2, 0.8]
                    lower = [A[2], 0]
                    upper = [A[end-1], 1]
                else
                    initial = A[end]/2
                    lower = A[2]
                    upper = A[end-1]
                end
                function Vf(x::Array{Float64,1};β = β, a = a,e=e,A=A,E=E)

                    return -(u(c(A[a],E[e],x[2],x[1]),1-x[2]) + β * EV(x[1],E[e]))
                end
                function Vf(x::Float64;β = β, a = a,e=e,A=A,E=E)
                    if E[e]<=0
                        return -(u(c(A[a],E[e],0.0,x),1.0) + β * EV(x,E[e]))
                    else
                        return -(u(c(A[a],E[e],1.0,x),0.0) + β * EV(x,E[e]))
                    end
                end


                if η!=1.0 && E[e]>0
                    maxV = optimize( Vf, lower,upper,initial,Fminbox(inner_optimizer))
                    policy[a,e,:] = maxV.minimizer
                elseif E[e]<=0
                    maxV = optimize( Vf,lower,upper)
                    policy[a,e,:] = [maxV.minimizer,0]
                else
                    maxV = optimize( Vf,lower,upper)
                    policy[a,e,:] = [maxV.minimizer,1]
                end
                Vgrid[a,e] = -maxV.minimum
            end
        end
        distance = maximum(abs.(Vgrid-Vgridf))
        #distance = maximum(abs.(policy-policy_old))
        V(a,e) = LinearInterpolation((A,E),Vgrid)(a,e)
        println("Distance is $(distance)")
    end
    #Finally, find labor, capital and consumption policies:
    policy_n(a,e) = LinearInterpolation((A,E),policy[:,:,2])(a,e)
    policy_a(a,e) = LinearInterpolation((A,E),policy[:,:,1])(a,e)
    policy_c(a,e) = c(a,e,policy_n(a,e),policy_a(a,e))
    return  policy_a,policy_n,policy_c,V
end
