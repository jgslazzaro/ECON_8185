
using Optim, Interpolations, Plots
include("CRRA_utility.jl")

#Uncomment the section below to test this code
#=
#Defining parameters
α = 0.3
β = 0.9
δ = 1
Z = (1-β*(1-δ))/(α * β) #This will normalize the SS to 1
ass = ((1- β*(1-δ))/(β*Z*α))^(1/(α-1))
η = 2/3
μ = 1


R(a,n)= 1#α*Z*a^(α-1)+(1-δ)
w(a,n) = 1.1

nA = 35 #number of Capital gridpoints



#productivity shocks
#E = [0,1]
#pdfE = [1/3 2/3;
#1/4 3/4]
E = [0.0,1.0]
pdfE =[3/4 1/4; 5/6 1/6]

nE = length(E)

#Asset grid:
A = range(-1,stop = 6, length = nA)
#A = zeros(nA)
#A[1:Int64(3/5*nA)] = range(-2.5,stop=ass,length=Int64(3*nA/5))
#A[Int64(3*nA/5)+1:end] = range(ass+2*ass/nA,stop=8*ass,length=Int64(2*nA/5)) =#


function VFI(A,E,pdfE;α = α,β = β, η = η, μ =μ,inner_optimizer = NelderMead(), tol = 1e-6)
    #Defining consumption function:
    nA = length(A)
    nE = length(E)
    c(a::Real,e::Real, n::Real ,a1::Real) =  R(a,n)*a+e*w(a,n)*n-a1

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

        Threads.@threads for a = 1:nA #Parallel for!!
            for e = 1:nE
                #solver stuff:
                if η != 1 && E[e]>0
                    initial = [1., 0.8]
                    lower = [A[2], 0]
                    upper = [A[end-1], 1]
                else
                    initial = 1.
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


                if η!=1 && E[e]>0
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
#@time policy_a, policy_n, policy_c, V = VFI(A,E,pdfE;α = α,β = β, η = η, μ =μ,inner_optimizer = BFGS())

#plot(A[1]:0.05:A[end],[policy_a.(A[1]:0.05:A[end],1) Z*α*β.*(A[1]:0.05:A[end]).^α A[1]:0.05:A[end]], label =["Approximation" "True" "45" ]) #log utility, exogenous labor case
#plot(A[1]:0.05:A[end],[policy_a.(A[1]:0.05:A[end],1) policy_a.(A[1]:0.05:A[end],0) A[1]:0.05:A[end]],label =["Employed", "Unemployed","45"],legend = :bottomright)

#plot(A,[policy_a.(A) Z*α*β.*(A).^α A])
