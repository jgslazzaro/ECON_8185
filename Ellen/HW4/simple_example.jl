#This code is to find the simple example:
#d'(x)+d(x)=0, x∈[0,xmax]
#The solution is d(x) = exp(-x)
using ForwardDiff,QuadGK, NLsolve,Plots, Optim,FastGaussQuadrature

cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Ellen\\HW4")

#Setting the grid for x

xmax = 10

X = [0 1 2 3 6]

#Piecewise linear Approximation:

#Constructing the basis function:

function ϕi(x,X,i::Int)
    if i>1 && i<length(X) #i is not in a boundary
        if X[i-1]<=x<=X[i]
            f = (x-X[i-1])/(X[i]-X[i-1])
        elseif X[i]<=x<=X[i+1]
            f = (X[i+1]-x)/(X[i+1]-X[i])
        else
            f = 0
        end
    elseif i==1
        if X[i]<=x<=X[i+1]
            f = (X[i+1]-x)/(X[i+1]-X[i])
        else
            f = 0
        end
    elseif i==length(X)
        if X[i-1]<=x<=X[i]
            f = (x-X[i-1])/(X[i]-X[i-1])
        else
            f=0
        end
    end
    return f
end


function derivϕi(x,X,i)
    if i>1 && i<length(X) #i is not in a boundary
        if X[i-1]<=x<=X[i]
            f = 1/(X[i]-X[i-1])
        elseif X[i]<=x<=X[i+1]
            f = -1/(X[i+1]-X[i])
        else
            f = 0
        end
    elseif i==1
        if X[i]<=x<=X[i+1]
            f = -1/(X[i+1]-X[i])
        else
            f = 0
        end
    elseif i==length(X)
        if X[i-1]<=x<=X[i]
            f = 1/(X[i]-X[i-1])
        else
            f=0
        end
    end
    return f
end

function residual(x,θ)
    R=0
    for i=1:length(X)
        R = R+ θ[i] *(derivϕi(x,X,i) + ϕi(x,X,i))
    end
    return R
end

#Minimizing the integral using non-linear method

function mini(α)
    if length(α) < length(X)
        α = vcat(1,α)
    end

nodes, weights = gausslegendre(200) #Gauss Legendre nodes and weights
    function integra(x)
    T=zeros(length(X))
        for i=1:length(X)
            T[i] = ϕi(x,X,i)*residual(x,α)
        end
    return T
    end
    #g = quadgk.(integra,X[1],X[end])[1] #Integral
    #See Judd's book pg 261 on numerical integration and the gausslegendre formula
    g = sum((X[end]-X[1])/2 .* weights .* integra.((nodes .+1).*(X[end]-X[1])/2 .+ X[1]))

    return sum(abs.(g))
end

#Minimizing the weighted integral
initial = ones(length(X)-1)
mini(initial)

bla = optimize(mini,initial,BFGS())#;autodiff = :forward)
θ = vcat(1,bla.minimizer)


#Using linear method as comparison
m=length(X)
K = zeros(m,m)

for i=1:m-1
    global K

    l = X[i+1]-X[i]
    Ki = [l/3-1/2 l/6+1/2; l/6-1/2 l/3+1/2]
    K[i:i+1,i:i+1] = K[i:i+1,i:i+1] + Ki
end

K
θlinear = K[2:end,2:end] \ (zeros(m-1)-K[2:end,1])

θlinear = vcat(1,θlinear)

difference = abs.(θ-θlinear)

#Get the solution:

function d(x)
    n=length(θ)
    solution =0
    for i =1:n
        solution = solution + θ[i]*ϕi(x,X,i)
    end
    return solution
end

x=0:0.1:6

plot(x,[exp.(-x) d.(x)])
