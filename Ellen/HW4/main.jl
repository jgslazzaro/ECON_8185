#Author: João Lazzaro
#Main code used to solve HW 4


using Plots, QuadGK, Optim, ForwardDiff, LinearAlgebra, FastGaussQuadrature
#Defining parameters
θ = 0.25
β = 0.9
δ = 1
A = (1-β*(1-δ))/(θ * β)

kss = ((1- β*(1-δ))/(β*A*θ))^(1/(θ-1))

function ϕi(x,X,i::Int) #Piecewise Linear function
    if i>1 && i<length(X) #i is not in a boundary
        if X[i-1]<=x<=X[i]
            f = (x-X[i-1])/(X[i]-X[i-1])
        elseif X[i]<=x<=X[i+1]
            f = (X[i+1]-x)/(X[i+1]-X[i])
        else
            f = 0
        end
    elseif i==1 #i is in the boundary(1)
        if X[i]<=x<=X[i+1]
            f = (X[i+1]-x)/(X[i+1]-X[i])
        else
            f = 0
        end
    elseif i==length(X) #i is in the top boundary
        if X[i-1]<=x<=X[i]
            f = (x-X[i-1])/(X[i]-X[i-1])
        else
            f=0
        end
    end
    return f
end


K = zeros(11)
for i=2:length(K)
    K[i] = K[i-1] + 0.005*exp(0.574*(i-2))
end

K = 1/4:1/8:2


function cn(k,K,α) #Defining consumption approximate function
    n = length(K)
    c = 0
    for i = 1:n
        c = c + α[i]*ϕi(k,K,i)
    end
    return c
end

polk(k,α) = max(0,A*k.^θ+(1-δ)*k-cn(k,K,α)) #capital policy function from Bugdet constraint

function residual(k,α) #Residual function comes from FOCs
    R =  (cn(k,K,α)*β*(A*θ*polk(k,α)^(θ-1)+1-δ))/cn(polk(k,α),K,α) - 1
    if R==-Inf
        R=-1e10
    elseif R==Inf
        R=1e10
    end
    return R
end

#This function calculates the weighted integral for a given parameter α
nodes, weights = gausslegendre(500) #Gauss Legendre nodes and weights

function mini(α)
    if length(α) < length(K)
        α = vcat(0,α)
    end
    function integra(k)
    T=zeros(length(K))
        for i=1:length(K)
            T[i] = ϕi(k,K,i)*residual(k,α)
        end
    return T
    end
    #g = quadgk.(integra,K[1],K[end])[1] #Integral
    #See Judd's book pg 261 on numerical integration and the gausslegendre formula:
    gaussleg = sum((K[end]-K[1])/2 .* weights .* integra.((nodes .+1).*(K[end]-K[1])/2 .+ K[1]))

    return norm(gaussleg,1)
end

#ForwardDiff.gradient(mini,initial)
#Minimizing the weighted integral
initial =  vcat(0,rand(length(K)-1))
α = initial
mini(initial)

bla = optimize(mini,initial, BFGS())

#global_min, α = minimise(mini)
α = vcat(0,bla.minimizer)
mini(α)

#Plotting
k=K[1]:0.01:K[end]

c(k) = (1-β*θ)*A*k^θ
cnplot(k) = cn(k,K,α)
plot(k,[cnplot.(k),c.(k)])
