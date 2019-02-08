using Optim, LinearAlgebra, FastGaussQuadrature


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

#Derivative of Finite elements Piecewise Linear function:
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

function derivcn(k,α;K=K)
#This function is piecewise linear approximation for the derivative of a function with arguments k
#To get a good approximations one need to find the parameters α
#K are the elements nodes. It is optionally defined here for efficiency.
        n = length(K)
        c = 0
        for i = 1:n
            c = c + α[i]*derivϕi(k,K,i)
        end
        return c
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
