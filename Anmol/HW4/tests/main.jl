include("functions.jl")
include("VFI_KS.jl")
#Defining parameters they were taken frmom KS
const α = 0.36
const β = 0.99
const δ = 0.025

const η = 1.0/2.9
const μ = 1.0

const N = 5000 #number of agents in each economy
const T = 11000 #number of simulation periods
const lbar = 1/0.9 #time endowment

nA = 8 #number of assets gridpoints
nK = 8 #Number of aggregate capital gridpoints
nH = 8 #Number of aggregate labor gridpoints

amin = eps()
amax = 100.0

#productivity shocks
Z = [0.99,1.01]
nZ=length(Z)

#Employment shocks
E = [0.0,1.0]
nE = length(E)

pdf = [0.525  0.35 0.03125 0.09375;
    0.038889 0.836111 0.002083 0.122917;
    0.09375 0.03125 0.291667 0.583333;
    0.009155 0.115885 0.024306 0.850694]

states = ([Z[1],E[1]],[Z[1],E[2]],[Z[2],E[1]],[Z[2],E[2]])

#Guessing Law of motions parameters
b = [0.114 0.953 ; 0.123 0.951]
d = [-0.592 -0.255;-0.544 -0.252]



#Asset grid:

factor = 3
A =(range(amin^(1/factor), stop=amax^(1/factor), length=nA)).^factor


H =range(eps(),stop = lbar, length = nH).^1

kssgood = exp(b[2,1]/(1-b[2,2]))
kssbad = exp(b[1,1]/(1-b[1,2]))
hssbad  = exp(d[1,1]/(1-d[1,2]))
hssgood = exp(d[2,1]/(1-d[2,2]))

kmin =  max(min(kssgood,kssbad)-5,eps())
kmax =  max(kssgood,kssbad)+10
K = range(kmin,stop = kmax, length = nK).^1
#K = range(30.0,stop = 50.0, length = nK)


@time b, d,  nsim, asim, Ksim, Hsim,policygrid,Vgrid,K,R2b,R2d,zsim,esim = KrusselSmith(A,E,Z,pdf,states,K,H,b,d;
iterate = "Value",N=N,T=T,verbose = false,discard = 1000, inner_optimizer = BFGS(linesearch = BackTracking(order=3)))#(linesearch = BackTracking(order=3)),lbar = lbar)



using JLD2, FileIO
@save "variables_nA$(nA).jld2" A E Z pdf K H states α β η μ lbar δ N T nA nE nH nK b d nsim asim Ksim Hsim policygrid Vgrid

#=
@load "variables.jld2"

using Plots#
plot1 = plot(Ksim, label = ["Aggregate Capital" ], legend = :bottomright)
savefig(plot1, "AggK.png")

plot2 = plot(Hsim, label = ["Aggregate Labor"], legend = :bottomright)
savefig(plot2, "AggH.png")


@save "variables.jld2" A E pdfE Z pdfZ K H K1 H1 policy_a policy_n policy_c V b d csim nsim asim Ksim Hsim α β δ η μ


#@load "variables.jld2"
=#
