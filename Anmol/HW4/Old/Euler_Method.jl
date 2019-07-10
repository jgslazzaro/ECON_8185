#This code finds the policy functions of the deterministic growth model using
#the Euler equation method.
#The idea is to iterate on the Euler Equation:
#u'(c)/(βu'(c')) = αAk'^(α-1) + 1-δ

#Let's first define the utility functions (CRRA)
using NLsolve, Optim, Interpolations
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
function uc(c,l;η = η,μ = μ)
    if η!=1 && l <= 0
        return 0.0
    elseif c<=0
        return 1e10*(-c) +(η*eps()^(η-1))*(eps()^η )^(-μ)
    else
        return (η * c^(η-1) * l^(1-η)) * (c^η * l^(1-η))^(-μ)
    end
end

#derivative of u with respect to l
function ul(c,l;η = η,μ = μ)
    if η!=1
        if c<=0
            return 0.0
        elseif l<=0
            return 1e10 * (-l) +1e5
        else
            return ((1-η) * c^η * l^(-η))*(c^η * l^(1-η))^(-μ)
        end
    else
        return 0.0
    end

end

#Now, we define the production function:
function f(k,l;δ = δ, A = A, α = α)
    if k<=0||l<=0
        return 0
    else
        return A*k^α * l ^(1-α) + (1-δ) * k
    end
end
function fk(k,l;δ = δ, A = A, α = α, η = η)
    if η!=1
        if  k<=0
            return 1e7 * (-k) +1e12
        elseif l<=0
            return 0.0
        else
            return A*α * k^(α-1) * l ^(1-α) + (1-δ)
        end
    else
        return A*α * k^(α-1) + (1-δ)
    end
end
function fl(k,l;δ = δ, A = A, α = α, η = η)
    if η !=1
        if  k<=0
            return 0
        elseif l<=0
            return 1e11 * (-l)+1e5
        else
            return A*(1-α) * k^(α) * l ^(-α)
        end
    else
        return A*(1-α) * k^(α)
    end
end

#Defining parameters
α = 0.33
β = 0.99
δ = 0.025
A = (1-β*(1-δ))/(α * β) #This will normalize the SS to 1
kss = ((1- β*(1-δ))/(β*A*α))^(1/(α-1))
η = 4/5
μ = 1

nK = 50 #number of Capital gridpoints


#Capital grid:

K= range(0.01,stop=2,length=nK)

#


function iterateEE(K; tol = 1e-6, η = η)
    #guessing policy functions
    gk0(k) = 0.5*k
    gl0(k) = 4/5
    gc0(k) = f(k,gl0(k)) -gk0(k)
    nK = length(K)
    k=1.0
    #defining the Euler Equation:
    function Eulereq(kl; k = k, gk = gk0, gl = gl0, η = η)

            k1 = kl[1]
            if η != 1
                l = kl[2]
                h = 1 - l #leisure
                h1 = 1 .- gl.(k1) #policy leisure
                consumptionFOC = (uc.(f.(k,l).-k1,h) .- β .* fk.(k1,gl.(k1)) .* uc.(f.(k1,gl.(k1)) - gk.(k1),h1))
                laborFOC = -(ul.(f.(k,l).-k1,h) .+ fl.(k,l) .* uc.(f.(k,l).-k1,h))
                F = vcat(consumptionFOC,laborFOC)
            else
                F = uc.(f.(k,1).-k1,0) .- β .* fk.(k1,1) .* uc.(f.(k1,1) - gk.(k1),0)
            end
            F=F.^2
            return maximum(F)
    end
    function Eulereq!(F,kl; k = k, gk = gk0, gl = gl0, η = η)

            k1 = kl[1]
            if η != 1
                l = kl[2]
                h = 1 - l #leisure
                h1 = 1 .- gl.(k1) #policy leisure
                consumptionFOC = (uc.(f.(k,l).-k1,h) .- β .* fk.(k1,gl.(k1)) .* uc.(f.(k1,gl.(k1)) - gk.(k1),h1))
                laborFOC = -(ul.(f.(k,l).-k1,h) .+ fl.(k,l) .* uc.(f.(k,l).-k1,h))
                F .= vcat(consumptionFOC,laborFOC)
            else
                F .= uc.(f.(k,1).-k1,0) .- β .* fk.(k1,1) .* uc.(f.(k1,1) - gk.(k1),0)
            end

    end

    #Interpolating policy functions:
    function g1(k,sol; K = K)
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
    solk = ones(nK)
    solk0 = ones(nK)
    soll0 = ones(nK).-1/3
    sol = ones(nK,2)
    #Iterating on Policy functions
    it =0
    euler = zeros(2)
    while d>tol
        #global gk0, gl0,it,d,sol,solk0,soll0,k
        if η!=1
          for i = 1:nK
              #global gk0, gl0,it,d,sol,solk0,soll0,k]
              sol[i,:] = optimize(Eulereq, [0.0,0.0], [5,1.0],[0.9,4/5],
              Fminbox(BFGS())).minimizer
              #sol[i,:] = nlsolve(Eulereq!, [solk0[i],soll0[i]], autodiff = :forward).zero
            end
            d = maximum(abs.(sol .- hcat( gk0.(K),gl0.(K))))
            solk0 = copy(sol[:,1])
            soll0 = copy(sol[:,2])
            gl0(k) =LinearInterpolation(K,soll0, extrapolation_bc=Line())(k)
            gk0(k) =LinearInterpolation(K,solk0, extrapolation_bc=Line())(k)
            #gl0(k) =g1(k,soll0; K = K)
            #gk0(k) =g1(k,solk0; K = K)

        else
            for i = 1:nK
                k = K[i]
                solk[i] = optimize(Eulereq, 0.0, 120,Brent()).minimizer
                #solk[i] = nlsolve(Eulereq!, [solk0[i]], autodiff = :forward).zero[1]

            end
            d = maximum(abs.(solk - gk0.(K)))
            solk0 = copy(solk)
            gk0(k) = LinearInterpolation(K,solk0, extrapolation_bc=Line())(k)
            #gk0(k) = g1(k,solk0; K = K)
            gl0(k) = 1.0
        end
        it +=1
        println("end of iteration $(it), distance is $(d)")

    end
    return gk0, gl0,gc0
end


#F=ones(2*length(K))
#F = Eulereq!(F,0.5 .* ones(2*length(K)); k = K, gk = gk0, gl = gl0)
#nlsolve(Eulereq!,0.2.*ones(2*length(K)), autodiff = :forward)


@time gk,gl,gc = iterateEE(K)
#gk, gl = gk0, gl

using Plots
#plot([0.1:0.01:K[end]],[gk.(0.1:0.01:K[end]),gl.(0.1:0.01:K[end]), A*α*β*(0.1:0.01:K[end]).^α],
#label = ["Capital Approximation" "Labor Approximation" "True"],legend=:bottomright)


plot([0.1:0.01:K[end]],[0.1:0.01:K[end],gk.(0.1:0.01:K[end])],label = ["45","Capital Approximation" ], legend = :bottomright)
#plot([0.1:0.01:2],gl.(0.1:0.01:2),label = ["Labor Approximation" ])
#plot([0.1:0.01:2],gc.(0.1:0.01:2),label = ["Consumption Approximation" ])
