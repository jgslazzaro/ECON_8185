
#cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
include("functions.jl")
using  LinearAlgebra, Plots
using  JLD2,FileIO

#Defining Parameters:

β = 0.98 #Discount rate
μ = 1.5  #Elasticity of intertemporal substitution
η = 1#Utility parameter
τy = 0.0 #Income tax
ρ = 0.6 #autocorrelation
σ = 0.3 #Variance
δ = 0.075 #Depreciation rate
θ = 0.3 #Capital Share
T=0

b = 0 #Debt limit
amax=2

nE = 5 #Number of states for e
nA = 500


r0= (1/β +δ)/2
K = ((r0+δ)/θ)^(1/(θ-1))
w0=  (1-θ)*K^θ

##

#Defining grids

pdfE,E = Tauchen(ρ,σ,nE)    #E
A = range(b,stop = amax, length = nA)    #Assets
policy_a, policy_c, policy_l = VFI(A,E,r0,w0)

@time λ,r,w, policy_a, policy_c, policy_l = ayiagary(A,E,r0,w0)

plot(sum(λ,dims=2))

@save "variables.jld2"  λ,r,w, policy_a, policy_c, policy_l


k = ayiagary_supply(A,E,r0,w0,100)

r_range=range(0.0,stop = 1.0,length = 100)
K = ((r_range.+δ)./θ).^(1/(θ-1))

@save "SupplyDemand.jld2"  r_range,K,k

plot(r_range,[K,k])
