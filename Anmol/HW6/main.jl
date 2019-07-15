
#This code is for Anmol HW6
using Optim, Interpolations
include("functions.jl")
#Defining parameters
const β = 0.9
const ψ = 0.69
const Π = 0.5 * ones(2, 2)
const G = [0.1, 0.2]

# Derivatives of utility function
U(c,n;ψ=ψ) = log(c) + ψ * log(1 - n)
Uc(c,n) = 1 ./ c
Un(c,n;ψ=ψ) = -ψ ./ (1.0 .- n)


nμ = 50
nB = 120
B = range(-1.0,stop=1.5,length = nB)
μ = range(0,stop = .55,length = nμ)

nΠ = length(Π)
nG = length(G)


#Guess for value function:


#Solve the saddle point problem:
Wgrid,policy = SPFE(B,G,μ;nB=nB,nG=nG,nμ=nμ)


#Simulate and plot the economy:
T = 20
Ghist = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
b0 = .50
μ0 = 0.0

Bhist = fill(findfirst(B.>=b0),T)
μhist = fill(findfirst(μ.==μ0),T)
nhist = ones(T)


for t = 2:T
    Bhist[t] = policy[Bhist[t-1],μhist[t-1],Ghist[t-1],1]
    μhist[t] = policy[Bhist[t-1],μhist[t-1],Ghist[t-1],2]
end

for t=1:T
    nhist[t] = nstar(μ[μhist[t]],B[Bhist[t]],G[Ghist[t]],μ[μhist[t]])
end

Bhist = B[Bhist]
Ghist = G[Ghist]
μhist = μ[μhist]

chist = nhist .-Ghist

tauhist = 1 .+ Un.(chist,nhist) ./ (1. * Uc.(chist,nhist))


using Plots


titles = hcat("Consumption","labor","Government Debt","Tax Rate","Government Spending", "Lagrange Multiplier")
p = plot(size = (920, 750), layout = grid(3, 2),
         xaxis=(0:T), grid=false, titlefont=Plots.font("sans-serif", 10))
plot!(p, title = titles, legend=false)
plot!(p[1], chist, marker=:utriangle, color=:blue)
plot!(p[2], nhist, marker=:utriangle, color=:blue)
plot!(p[3], Bhist, marker=:utriangle, color=:blue)
plot!(p[4], tauhist, marker=:utriangle, color=:blue)
plot!(p[5], Ghist, marker=:utriangle, color=:blue)
plot!(p[6], μhist, marker=:utriangle, color=:blue)



#3 values non iid:

const Π = [1/6 3/6 2/6; 1/6 3/6 2/6; 1/6 2/6 3/6]
const G = [0.05, 0.1, 0.2]
nμ = 50
nB = 120
B = range(-1.0,stop=1.5,length = nB)
μ = range(0,stop = .55,length = nμ)

nΠ = length(Π)
nG = length(G)


#Solve the saddle point problem:
Wgrid,policy = SPFE(B,G,μ;nB=nB,nG=nG,nμ=nμ)


#Simulate and plot the economy:
T = 200
Ghist = ones(Int64,T)*2
for t = 2:T
    global Ghist
    g0 = findfirst(G.==G[Ghist[t-1]])
    s = rand()

    g1 = 1
    p = Π[g0,1]
    while s>p
        g1 += 1
        p += Π[g0,g1]
    end
    Ghist[t] = g1
end
t=2
Ghist



b0 = .50
μ0 = 0.0

Bhist = fill(findfirst(B.>=b0),T)
μhist = fill(findfirst(μ.==μ0),T)
nhist = ones(T)


for t = 2:T
    Bhist[t] = policy[Bhist[t-1],μhist[t-1],Ghist[t-1],1]
    μhist[t] = policy[Bhist[t-1],μhist[t-1],Ghist[t-1],2]
end

for t=1:T
    nhist[t] = nstar(μ[μhist[t]],B[Bhist[t]],G[Ghist[t]],μ[μhist[t]])
end

Bhist = B[Bhist]
Ghist = G[Ghist]
μhist = μ[μhist]

chist = nhist .-Ghist

tauhist = 1 .+ Un.(chist,nhist) ./ (1. * Uc.(chist,nhist))
tauhist[1] =0.4

using Plots


titles = hcat("Consumption","labor","Government Debt","Tax Rate","Government Spending", "Lagrange Multiplier")
p = plot(size = (920, 750), layout = grid(3, 2),
          grid=false, titlefont=Plots.font("sans-serif", 10))
plot!(p, title = titles, legend=false)
plot!(p[1], chist, marker=:utriangle, color=:blue)
plot!(p[2], nhist, marker=:utriangle, color=:blue)
plot!(p[3], Bhist, marker=:utriangle, color=:blue)
plot!(p[4], tauhist, marker=:utriangle, color=:blue)
plot!(p[5], Ghist, marker=:utriangle, color=:blue)
plot!(p[6], μhist, marker=:utriangle, color=:blue)
