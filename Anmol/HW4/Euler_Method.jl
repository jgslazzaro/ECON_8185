#This code finds the policy functions of the deterministic growth model using
#the Euler equation method.
#The idea is to iterate on the Euler Equation:
#u'(c)/(βu'(c')) = αAk'^(α-1) + 1-δ

#Let's first define the utility functions (CRRA)

function u(c,l;η = η,μ = μ)
    if (c<=0) || (η<1 && l==0)
        u=-Inf
    elseif μ == 1
        u = log(c^η * l^(1-η))
    else
        u = ((c^η * l^(1-η))^(1-μ) )/ (1-μ)
    end
    return u
end

#derivative of u with respect to c
uc(c,l;η = η,μ = μ)  = (η * c^(η-1) * l^(1-η)) * (c^η * l^(1-η))^(-μ)

#derivative of u with respect to l
ul(c,l;η = η,μ = μ) = ((1-η) * c^η * l^(-η))*(c^η * l^(1-η))^(-μ)

#Now, we define the production function:
f(k,l;δ = δ, A = A, α = α) = A*k^α * l ^(1-α) + (1-δ) * k
fk(k,l;δ = δ, A = A, α = α) = A*α * k^(α-1) * l ^(1-α) + (1-δ)
fl(k,l;δ = δ, A = A, α = α) = A*(1-α) * k^(α) * l ^(-α) + (1-δ)

#Defining parameters
α = 0.3
β = 0.9
δ = 1
A = (1-β*(1-δ))/(α * β) #This will normalize the SS to 1
kss = ((1- β*(1-δ))/(β*A*α))^(1/(α-1))
η = 1
μ = 1

nK = 15 #number of Capital gridpoints

#Capital grid:
K=zeros(nK)
K[1:Int64(2*nK/3)] = range(0.1 * kss,stop=kss,length=Int64(2*nK/3))
K[Int64(2*nK/3)+1:end] = range(kss+2/nK,stop=2,length=Int64(1*nK/3))


using NLsolve
function iterateEE(K; tol = 1e-7)
    #defining the Euler Equation:
    function Eulereq!(F,k1; k = K, gk = gk0)
            F .= uc.(f.(k,1).-k1,1)./uc.(f.(k1,1) - gk.(k1),1) - β .* fk.(k1,1)
    end
    #guessing policy function
    gk0(k) = 0.9*k
    gl0(k) = 1.0
    function gk1(k,sol; K = K)
            k1 = 0.0
            for i = 1:length(K)
                if k == K[i]
                    k1 = sol[i]
                    break
                elseif i<length(K) && K[i]< k <K[i+1]
                    k1 = sol[i] + (k - K[i]) * (sol[i+1] - sol[i]) / (K[i+1] - K[i])
                    break
                end
            end
            return k1
    end

    d=1.0
    sol = zeros(length(K))
    sol0 = zeros(length(K))
    while d>tol
        sol = nlsolve(Eulereq!, kss.*ones(length(K)), autodiff = :forward).zero
        d = maximum(abs.(sol - gk0.(K)))
        sol0 = copy(sol)
        gk0(k) = gk1(k,sol0; K = K)
    end
    return gk0
end



gk = iterateEE(K; tol = 1e-7)
using Plots
plot([0.1:0.01:2,0.1:0.01:2],[gk.(0.1:0.01:2), A*α*β*(0.1:0.01:2).^α], label = ["Approximation" "True"])
