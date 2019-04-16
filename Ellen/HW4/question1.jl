

using Plots

#Question 1
#Defining parameters
θ = 0.25
β = 0.9
δ = 1
A = (1-β*(1-δ))/(θ * β) #This will normalize the SS to 1
kss = ((1- β*(1-δ))/(β*A*θ))^(1/(θ-1))


#Defining the elements:
K = zeros(15)
for i=2:length(K)
    global K
    K[i] = K[i-1] +0.0005*exp(0.574*(i-2))
end

#load the functions compatible with ANY finite elements methods
include("finite_elements_functions.jl")


#capital policy function from Bugdet constraint
polk(k,α) = min(max(eps(),A*k.^θ+(1-δ)*k-cn(k,α)),K[end])
#min max are needed to avoid NaNs and other numerical instabilities

function residual(k,α)
    #This function is specific for the discrete time deterministic growth model.
    #Residual function comes from FOCs
    #cn below is an approximation for consumption
        R = cn(k,α)/cn(polk(k,α),α) * β * (A*θ*polk(k,α)^(θ-1)+1-δ)- 1
    return R
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
polkplot(k) = polk(k,α)
cnplot(k) = cn(k,α)
plot(k,[cnplot.(k),c.(k)],label=["Approximation" "True Function"],legend=:bottomright)
savefig("question1.png")

#Uncomment below to get δ=0.05
#=
#New parameters
δ = 0.05
A = (1-β*(1-δ))/(θ * β) #This will normalize the SS to 1

kss = ((1- β*(1-δ))/(β*A*θ))^(1/(θ-1))

polk(k,α) = min(max(eps(),A*k.^θ+(1-δ)*k-cn(k,α)),K[end]) #capital policy function from Bugdet constraint

function residual(k,α) #Residual function comes from FOCs
        R = β * cn(k,α)/cn(polk(k,α),α)  * (A*θ*polk(k,α)^(θ-1)+1-δ)- 1
    return R
end

initial = ones(length(K)-1)*0.0035
for i=2:length(initial)
    global initial
    initial[i] = initial[i-1] +0.000265*exp(0.589*(i-2))
end
initial
#initial = ones(length(K)-1)*0.1.*range(0.03,stop = 8,length = length(K)-1)
mini(initial)

#Solver stuff
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

cn2plot(k) = cn(k,α)
plot(k,[cn2plot.(k)],label=["Approximation"],legend=:bottomright)
savefig("question1b.png") =#
