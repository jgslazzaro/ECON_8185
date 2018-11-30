
using  LinearAlgebra, Plots
using  JLD2,FileIO
include("functions.jl")


#Defining Parameters:
β = 0.98 #Discount rate
μ = 1.5  #Elasticity of intertemporal substitution
η = 1#Utility parameter
τy = 0.0 #Income tax
ρ = 0.95 #autocorrelation
σ = 0.3 #Variance
δ = 0.075 #Depreciation rate
θ = 0.3 #Capital Share
T=0
Z=1

b = -3#Debt limit
amax=5

nE = 2 #Number of states for e
nA = 20 #states for assets


r= 0.7#(1/β-1)*rand()
K = ((r0+δ)/(Z*θ))^(1/(θ-1))
w=  (1-θ)*K^θ

##



#Defining grids
pdfE,E = Tauchen(ρ,σ,nE)    #E
A = range(b,stop = amax, length = nA)
u = ones(nA,nE,nA)
policy_l1=ones(nA,nE,nA)

#Solving the labor problem for each level of asset today and tommorow
#I tested and it is faster than including this in the main while loop
c(l,a,e,a1) = (1-τy)*E[e]*w*(1-l)+T+(1+(1-τy)*r)*A[a]-A[a1]
for a=1:nA,e=1:nE,a1=1:nA
    maxu(l) = -utility(c(l,a,e,a1),l,η,μ)
    maxU = optimize(maxu, 0.0, 1.0)
    u[a,e,a1] = -maxU.minimum
    policy_l1[a,e,a1] = maxU.minimizer
end

#VFI
#Initial value function guess:
V  = ones(nA,nE)
#preallocation
policy_a = Array{Int64,2}(undef,nA,nE)
#loop stuff
distance = 1
tol =10^(-7)
#iterating on Value functions:
while distance >= tol
    global distance, tol, V, policy_a
    Vf = copy(V) #save V to compare later
    #find the expected Vf: E[V(k,z')|z] = π(z1|z)*V(k,z1) +π(z2|z)*V(k,z2)
    EVf= (pdfE*Vf')'  #The transposing fix the dimensions so it works!
    for a in 1:nA, e in 1:nE
        V[a,e] , policy_a[a,e] = findmax(u[a,e,:] .+ β * EVf[:,e]) #Bellman Equation
        if u[a,e,policy_a[a,e]] == -Inf
            policy_a[a,e] = 1
        end
    end
 distance = maximum(abs.(V-Vf))
end
#Finally, find labor policy:
policy_l = Array{Float64,2}(undef,nA,nE)
for a in 1:nA, e in 1:nE
    policy_l[a,e] = policy_l1[a,e,policy_a[a,e]]
end

#and for consumption
policy_c=ones(nA,nE)
for a=1:nA,e=1:nE
    policy_c[a,e] = (1-τy)*E[e]*w*(1-policy_l[a,e])+T+(1+(1-τy)*r)*A[a]-A[policy_a[a,e]]
end
