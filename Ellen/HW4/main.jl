#Author: João Lazzaro
#Main code used to solve HW 4


using Plots, Optim, LinearAlgebra, FastGaussQuadrature

#Defining parameters
θ = 0.25
β = 0.9
δ = 1
A = (1-β*(1-δ))/(θ * β) #This will normalize the SS to 1
kss = ((1- β*(1-δ))/(β*A*θ))^(1/(θ-1))

#Finite elements Piecewise Linear function:
function ϕi(x,X,i::Int)
    #x: point to evaluate the function
    #X: Vector with elements nodes
    #i: Which element in the function
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

#Defining the elements:
K = zeros(15)
for i=2:length(K)
    global K
    K[i] = K[i-1] +0.0005*exp(0.574*(i-2))
end
K

#Defining consumption approximate function, this definition would hold for any
#function to be approximated
function cn(k,α;K=K)
#This function is piecewise linear approximation for a function with arguments k
#To get a good approximations one need to find the parameters α
#K are the elements nodes. It is optionally defined here for efficiency.
        n = length(K)
        c = 0
        for i = 1:n
            c = c + α[i]*ϕi(k,K,i)
        end
        return c
end

#capital policy function from Bugdet constraint
polk(k,α) = min(max(eps(),A*k.^θ+(1-δ)*k-cn(k,α)),K[end])
#min max are needed to avoid NaNs and other numerical instabilities

function residual(k,α)
    #This function is specific for the deterministic growth model.
    #Residual function comes from FOCs
    #cn below is an approximation for consumption
        R = cn(k,α)/cn(polk(k,α),α) * β * (A*θ*polk(k,α)^(θ-1)+1-δ)- 1
    return R
end

function integra(k,α;K=K)
#This function calculates the function that will be integrated:
#integra(k;α):= ϕi(k)R(k;α), where ϕi are the weights and R is the residual function
#In the Finite element methods, the weights are the same as the approximating functions
    T=zeros(length(K))
    for i=1:length(K)
        T[i] = ϕi(k,K,i)*residual(k,α)
    end
    return T
end

#This function calculates the integral (the norm of the integrated functions), as a functions of the parameters to minimized
#We define that way since this is the format accepted by the solver:
#mini(α):= ∫integra(k;α)dk
nodes, weights = gausslegendre(3*(length(K)-1)) #Gauss Legendre nodes and weights,this function is just a Quadrature table
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

#Setting initial conditions
initial =  ones(length(K)-1) .* range(0.35, stop = 3.5, length = length(K)-1)

#Check if the initial conditions are somewhat close to the true parameters.
mini(initial)

#Here we start the minimization procedure we want to find α:= argmin mini(α)

#Solver stuff:
#lower and upper bound of the parameters:
lower = zeros(length(initial))
upper = Inf*ones(length(initial))
#Optimizer method is BFGS, see Judd's book page 114 for an explanation:
inner_optimizer = BFGS()

#Solver:
bla = optimize(mini,lower,upper,initial, Fminbox(inner_optimizer))
#Parameters
α = vcat(0,bla.minimizer)

#Plotting
k=K[1]:0.01:K[end]

c(k) = (1-β*θ)*A*k^θ
cnplot(k) = cn(k,α)
plot(k,[cnplot.(k),c.(k)],label=["Approximation" "True Function"],legend=:bottomright)
