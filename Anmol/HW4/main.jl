using Distributions, Random
using Optim, Interpolations
using JLD2, FileIO

include("functions.jl")
include("VFI_KS2.jl")
#Defining parameters they were taken frmom KS
α = 0.33
β = 0.9
δ = 1.0

η = 1.0#1.9/2.9
μ = 1.0

N = 50 #number of agents in each economy
T = 750 #number of simulation periods



nA = 5 #number of assets gridpoints
nK = 2 #Number of aggregate capital gridpoints
nH = 2 #Number of aggregate labor gridpoints

#productivity shocks
Z = [0.99,1.01]
pdfZ = [1/2 1/2; 1/2 1/2]

#Constructing the stochastic pdf
pdf = [πz*πe for πz in pdfZ, πe in pdfE] #nZxnZxnExnE
pdf = permutedims(pdf, (1,3,2,4)) #nZxnExnZxnE


nZ=length(Z)

#Employment shocks
E = [0.0,1.0]
pdfE =[1/2 1/2; 1/2 1/2]
nE = length(E)

Npdf = [πz*πe for πe in pdfE, πz in pdfZ] #nExnExnZxnZ



#Asset grid:
A = range(eps(),stop = 15, length = nA)
K = range(A[1],stop = A[end], length = nK)
H = range(eps(),stop = 1, length = nH)


#Wages functions]
R(K::Float64,H::Float64,z::Float64;α::Float64 = α,δ::Float64 = δ)= z*α*K^(α-1)*H^(1-α) + (1-δ)
w(K::Float64,H::Float64,z::Float64;α::Float64 = α) = z*(1-α)*K^(α)*H^(-α)



#Guessing Law of motions
b = [0.0 1 ; 0 1]#[0.114 0.953 ; 0.123 0.951]
d = [0.0 0.0;0.0 0.0]#[-0.592 -0.255;-0.544 -0.252]
K1(K::Float64,z::Float64;b::Array{Float64,2}=b,Z::Array{Float64,1} = Z) = exp(b[findfirst(Z.==z),1]+b[findfirst(Z.==z),2]*log(K))
H1(K::Float64,z::Float64;d::Array{Float64,2}=d,Z::Array{Float64,1}=Z) = exp(d[findfirst(Z.==z),1]+d[findfirst(Z.==z),2]*log(K))

#@time K1(3.0,1.01)

#@time H1(3.0,1.01)


@time KrusselSmith(A,E,pdfE,Z,pdfZ,K,H,K1,H1,b,d; iterate = "Value",N=N,T=T,damp=2/3,damp2 = .5,verbose = true)

#policy_a,policy_n,policy_c,V,b,d, csim,nsim,asim,Ksim,Hsim,policygrid = KrusselSmith(A,E,pdfE,Z,pdfZ,K,H,K1,H1,b,d;
 #iterate = "Value",N=N,T=T,damp=2/3,damp2 = .5,verbose = true)

#=

using Plots
plot1 = plot(Ksim, label = ["Aggregate Capital" ], legend = :bottomright)
savefig(plot1, "AggK.png")

plot2 = plot(Hsim, label = ["Aggregate Labor"], legend = :bottomright)
savefig(plot2, "AggH.png")


@save "variables.jld2" A E pdfE Z pdfZ K H K1 H1 policy_a policy_n policy_c V b d csim nsim asim Ksim Hsim α β δ η μ


#@load "variables.jld2"
=#
