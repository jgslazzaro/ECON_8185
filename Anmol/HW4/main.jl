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
const lbar = 1.0/0.9 #time endowment

nA = 10 #number of assets gridpoints
nK = 5 #Number of aggregate capital gridpoints
nH = 5 #Number of aggregate labor gridpoints

amin = 1e-5
amax = 1000.0

ug=0.04 #Unemployment rate in good times
ub=0.1 #Unemployment rate in bad times
zg_ave_dur=8 #average duration of good period
zb_ave_dur=8 #average duration of bad period
ug_ave_dur=1.5 #average duration of unemployment period in good times
ub_ave_dur=2.5 #average duration of unemployment period in bad times
puu_rel_gb2bb=1.25 #imposed conditions
puu_rel_bg2gg=0.75

#productivity shocks
Z = [0.99,1.01]
nZ=length(Z)

#Employment shocks
E = [0.0,1.0]
nE = length(E)


tmat = create_transition_matrix()
transmat = tmat.P

states = ([Z[1],E[1]],[Z[1],E[2]],[Z[2],E[1]],[Z[2],E[2]])

#Guessing Law of motions parameters
b = [0. 1.0 ; 0. 1.0]
d = [log(lbar) -0.;log(lbar) -0.]

#Asset grid:
factor = 7
A =(range(0.0, stop=nA-1, length=nA)/(nA-1)).^factor * amax;
H =range(eps(),stop = lbar, length = nH).^1
K = range(30,stop = 50.0, length = nK).^1


@time b, d,  nsim, asim, Ksim, Hsim,policygrid,Vgrid,K,R2b,R2d,zsim,esim = KrusselSmith(A,E,Z,tmat,states,K,H,b,d;
N=N,T=T,discard = 1000, inner_optimizer = NelderMead())#(linesearch = BackTracking(order=3)),lbar = lbar)


policygrid

mean(Ksim)
mean(Hsim)
mean(esim)


using JLD2, FileIO
@save "variables_nA$(nA).jld2" A E Z transmat K H states α β η μ lbar δ N T nA nE nH nK b d nsim asim Ksim Hsim policygrid Vgrid

#=
@load "variables.jld2"

using Plots#
plot1 = plot(Ksim, label = ["Aggregate Capital" ], legend = :bottomright)
savefig(plot1, "AggK.png")

plot2 = plot(Hsim, label = ["Aggregate Labor"], legend = :bottomright)
savefig(plot2, "AggH.png")


@save "variables.jld2" A E transmatE Z transmatZ K H K1 H1 policy_a policy_n policy_c V b d csim nsim asim Ksim Hsim α β δ η μ


#@load "variables.jld2"
=#
