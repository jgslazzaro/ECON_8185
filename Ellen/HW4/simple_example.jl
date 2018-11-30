#This code is to find the simple example:
#d'(x)+d(x)=0, x∈[0,xmax]
#The solution is d(x) = exp(-x)
using ForwardDiff,QuadGK, NLsolve

cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Ellen\\HW4")

#Setting the grid for x

xmax = 10

X = [0 1.0 3 6]
θ = [1 2 1 5.0]
#Piecewise linear Approximation:

#Constructing the basis function:

function ϕ(x,X=X)

        if X[1]<=x<=X[2]
            f = (x-X[1])/(X[2]-X[1])
        elseif X[2]<=x<=X[3]
            f = (X[3]-x)/(X[3]-X[2])
        else
            f = 0
        end
    return f
end

function derivϕ(x,X=X)
        if X[1]<=x<=X[2]
            f = 1/(X[2]-X[1])
        elseif X[2]<=x<=X[3]
            f = -1/(X[3]-X[2])
        else
            f = 0
        end
    return f
end


function d(x,θ=θ,X=X)
    n=length(X)
    f = 0
    for i = 1:n
        if i==1
            f = f+ θ[i]*ϕ(x,[X[i] X[i] X[i+1]])
        elseif i==n
            f = f+ θ[i]*ϕ(x,[X[i-1] X[i-1] X[i]])
        else
                f=f+θ[i] *ϕ(x,X[i-1:i+1])
        end
    end
    return f
end



d(2.5)

function R(x,θ=θ,X=X)
        n=length(X)
        f = 0
        for i = 1:n
            if i==1
                f = f+ θ[i]*(derivϕ(x,[X[i] X[i] X[i+1]])+ϕ(x,[X[i] X[i] X[i+1]]))
            elseif i==n
                f = f+ θ[i]*(derivϕ(x,[X[i-1] X[i-1] X[i]])+ϕ(x,[X[i-1] X[i-1] X[i]]))
            else
                    f=f+θ[i] *(derivϕ(x,X[i-1:i+1])+ϕ(x,X[i-1:i+1]))
            end
        end
        return f
    end

R(0.4)

nlsolve(essa!,[1,4.7,1,0.95],ftol = :1.0e-9, method = :trust_region , autoscale = true)


μ
