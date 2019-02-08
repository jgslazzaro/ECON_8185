#Author: João Lazzaro
#Main code used to solve HW 4

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

#######################################################################################
#Stuff for question 2
using Plots

#Defining parameters
θ = 0.25
ρ = -log(0.9)
δ = 1
A = (δ + ρ)/θ  #This will normalize the SS to 1
kss = ((δ+ρ)/(A*θ))^(1/(θ-1))


#Defining the elements:
K = zeros(15)
for i=2:length(K)
    global K
    K[i] = K[i-1] +0.0005*exp(0.574*(i-2))
end
K

#load the functions compatible with ANY finite elements methods
include("finite_elements_functions.jl")

#Defining the residual function from Euler equation:
function residual(k,α)
    #This function is specific for the deterministic growth model.
    #Residual function comes from FOCs
    #cn is an approximation for consumption
    #derivcn is an approxiamtion for the derivative of consumption.
        R = θ*A*k^(θ-1)-(δ+ρ) - derivcn(k,α)/cn(k,α) * (A*k^θ-δ*k-cn(k,α))
    return R
end


#Setting initial conditions

initial = ones(length(K)-1)*0.05
for i=2:length(initial)
    global initial
    initial[i] = initial[i-1] +0.165*exp(0.08*(i-2))
end
initial

#initial =  ones(length(K)-1) .* range(0.3, stop = 5, length = length(K)-1)
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
αcont = vcat(0,bla.minimizer)

#Plotting
k=K[1]:0.01:K[end]

#c(k) = (1-β*θ)*A*k^θ
contplot(k) = cn(k,αcont)
plot(k,[contplot.(k)],label=["Continuous time Approximation"],legend=:bottomright)
savefig("question2.png")


#######################################################################################
#Stuff for question 3
#FIRST RUN THE STUFF FOR QUESTION 1 and 2!!!!!!!!

#Get the VFI objects, see HW1 or the file "run_VFI.jl" to adjust parameters.
include("run_VFI.jl")

#Plotting the consumption policies
plot(k,[cnplot.(k),c.(k),contplot.(k),policy_c],label=["Finite Elements" "True Function" "Continuous time" "VFI (150 gridpoints)"],legend=:bottomright)


#######################################################################################
#Stuff for question 4
#FIRST RUN THE STUFF FOR QUESTION 1 BUT NOT 2!!!!!!!!


include("run_VFI.jl")
k_finiteelements=0.5*ones(10)
k_VFI_index = Int(floor(KVFI/4))*ones(Int,10)
true_k =  0.5*ones(10)

for i=2:10
    global k_finiteelements, k_VFI_index
    k_finiteelements[i] = polkplot(k_finiteelements[i-1])
    k_VFI_index[i] = policy_k[k_VFI_index[i-1]]
    true_k[i] = β*θ*A*true_k[i-1]^θ
end
c_VFI = policy_c[k_VFI_index]
c_finiteelements = cnplot.(k_finiteelements)
true_c = c.(true_k)


plot(1:10,[c_finiteelements,true_c,c_VFI],label=["Finite Elements" "True Function" "VFI (150 gridpoints)"],legend=:bottomright)
