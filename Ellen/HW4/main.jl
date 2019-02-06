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

K=0:0.5:2

#Defining consumption approximate function
function cn(k,α;K=K)
        n = length(K)
        c = 0
        for i = 1:n
            c = c + α[i]*ϕi(k,K,i)
        end
        return c
    end

polk(k,α) = min(max(eps(),A*k.^θ+(1-δ)*k-cn(k,α)),K[end]) #capital policy function from Bugdet constraint

function residual(k,α) #Residual function comes from FOCs
        R = cn(k,α)/cn(polk(k,α),α) * β * (A*θ*polk(k,α)^(θ-1)+1-δ)- 1
    return R
end

#This function calculates the weighted integral for a given parameter α
function integra(k,α;K=K)
    T=zeros(length(K))
    for i=1:length(K)
        T[i] = ϕi(k,K,i)*residual(k,α)
    end
    return T
end

nodes, weights = gausslegendre(3*length(K)) #Gauss Legendre nodes and weights
function mini(α;nodes=nodes,weights=weights,K=K)
    if length(α)<length(K)
        α = vcat(0,α)
    end
    #g = quadgk.(integra,K[1],K[end])[1] #Integral
    #See Judd's book pg 261 on numerical integration and the gausslegendre formula:
    gaussleg = zeros(length(K))
    for j=1:length(nodes)
        gaussleg .+= (K[end]-K[1])/2 .* weights[j] .* integra((nodes[j] .+1).*
        (K[end]-K[1])/2 .+ K[1],α)
    end
    return norm(gaussleg,1)
end

#Minimizing the weighted integral

initial =  ones(length(K)-1) .* range(1, stop = 4, length = length(K)-1)
lower = zeros(length(initial))
upper = Inf*ones(length(initial))
inner_optimizer = BFGS()



grad!(A,initial)

@time mini(initial)

bla = optimize(mini,lower,upper,initial, Fminbox(inner_optimizer))


α = vcat(0,bla.minimizer)

mini(α)

#Plotting
k=K[1]:0.01:K[end]

c(k) = (1-β*θ)*A*k^θ
cnplot(k) = cn(k,α)
plot(k,[cnplot.(k),c.(k)])
