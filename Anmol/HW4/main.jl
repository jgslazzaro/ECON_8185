
using Distributions, Plots,Random
using Optim, Interpolations
using JLD2, FileIO

include("functions.jl")
include("VFI_KS.jl")
#Defining parameters they were taken frmom KS
α = 0.33
β = 0.9
δ = 1#0.025

η = 1#1.9/2.9
μ = 1

N = 500 #number of agents in each economy
T = 1500 #number of simulation periods



nA = 8 #number of assets gridpoints
nK = 2 #Number of aggregate capital gridpoints
nH = 2 #Number of aggregate labor gridpoints

#productivity shocks
Z = [0.99,1.01]
pdfZ = [1/8 7/8; 1/8 7/8]

nZ=length(Z)

#Employment shocks
E = [0.0,1.0]
pdfE =[1/8 7/8; 1/8 7/8]
nE = length(E)

#Asset grid:
A = range(eps(),stop = 5.3, length = nA)
K = range(A[1],stop = A[end], length = nK)
H = range(eps(),stop = 1, length = nH)


#Wages functions]
R(K,H,z)= z*α*K^(α-1)*H^(1-α) + (1-δ)
w(K,H,z) = z*(1-α)*K^(α)*H^(-α)

#Guessing Law of motions
b = [0.114,0.953;0.123,0.951]
d = [-0.592,-0.255;-0.544,-0.252]
K1(K,z::Int64;b=b) = exp(b[z,1]+b[z,2]*log(K))
H1(K,z::Int64;d=d) = exp(d[z,1]+d[z,2]*log(K))

policy_a,policy_n,policy_c,V,b,d, csim,nsim,asim,Ksim,Hsim,policygrid = KrusselSmith(A,E,pdfE,Z,pdfZ,K,H,K1,H1,b,d;
inner_optimizer = BFGS(), iterate = "Value",N=N,T=T,damp=2/3,damp2 = 1.0)



plot1 = plot(Ksim, label = ["Aggregate Capital" ], legend = :bottomright)
savefig(plot1, "AggK.png")

plot2 = plot(Hsim, label = ["Aggregate Labor" ], legend = :bottomright)
savefig(plot2, "AggH.png")


@save "variables.jld2" A E pdfE Z pdfZ K H K1 H1 policy_a policy_n policy_c V b d csim nsim asim Ksim Hsim α β δ η μ


#@load "variables.jld2"
=#
