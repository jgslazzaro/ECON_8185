
using Distributions, Plots,Random
using Optim, Interpolations
include("functions.jl")
include("VFI_KS.jl")
#Defining parameters they were taken frmom KS
α = 0.33
β = 0.99
δ = 0.025

η = 1.9/2.9
μ = 1

N = 5000 #number of agents in each economy
T = 10000 #number of simulation periods



nA = 50 #number of assets gridpoints
nK = 15 #Number of aggregate capital gridpoints
nH = 15 #Number of aggregate labor gridpoints

#productivity shocks
Z = [0.99,1.01]
pdfZ = [1/2 1/2; 1/2 1/2]

nZ=length(Z)

#Employment shocks
E = [0.0,1.0]
pdfE =[1/4 3/4; 1/6 5/6]
nE = length(E)

#Asset grid:
A = range(eps(),stop = 6, length = nA)
K = range(A[1],stop = A[end], length = nK)
H = range(eps(),stop = 1, length = nH)


#Wages functions
R(K,H,z)= z*α*K^(α-1)*H^(1-α) + (1-δ)
w(K,H,z) = z*(1-α)*K^(α)*H^(-α)

#Guessing Law of motions
b = [0,1.0]
d = [0.0,0]
K1(K;b=b) = exp(b[1]+b[2]*log(K))
H1(K;d=d) = exp(d[1]+d[2]*log(K))


policy_a,policy_n,policy_c,V,b,d, csim,nsim,asim,Ksim,Hsim = KrusselSmith(A,E,pdfE,Z,pdfZ,K,H,K1,H1,b,d;
inner_optimizer = BFGS(), iterate = "Policy",N=N,T=T)
plot(Ksim)
