include("functions.jl")
include("Carrol_KS.jl")
#Defining parameters they were taken frmom KS
const α = 0.36 #capital share in output
const β = 0.99 #Impatience
const δ = 0.025 #Depreciation rate

const η = 1.0/2.9 #labor /consumption elasticity
const μ = 1.0 #Intratemporal elasticity

const N = 5000 #number of agents in each economy
const T = 8000 #number of simulation periods
const lbar = 1.0#/0.9 #time endowment

const nA = 100 #number of assets gridpoints
const nK = 8 #Number of aggregate capital gridpoints
const nH = 8    #Number of aggregate labor gridpoints

amin = 1e-5
amax = 500.0

ug=0.04 #Unemployment rate in good times
ub=0.1 #Unemployment rate in bad times
zg_ave_dur=8 #average duration of good period
zb_ave_dur=8 #average duration of bad period
ug_ave_dur=1.5 #average duration of unemployment period in good times
ub_ave_dur=2.5 #average duration of unemployment period in bad times
puu_rel_gb2bb=1.25 #imposed conditions following KS
puu_rel_bg2gg=0.75 #imposed conditions

#productivity shocks
const Z = [0.99,1.01]
const nZ=length(Z)

#Employment shocks
const E = [0.0,1.0]
const nE = length(E)


const tmat = create_transition_matrix()
const transmat = tmat.P

const states = ([Z[1],E[1]],[Z[1],E[2]],[Z[2],E[1]],[Z[2],E[2]])

#Guessing Law of motions parameters
b = [0.114 0.953;0.123 0.951]#[0.0 1.0;0.0 1.0]#
d = [-0.592 -0.255;-0.544 -0.252]#[-0.4 -0.24;-0.5 -0.238]#
#Asset grid:
factor = 3.5
const A =(range(0, stop=nA-1, length=nA)/(nA-1)).^factor * amax #Capital grid for today
const A1 = A# range(0., stop=amax, length=nA).^1 #Capital grid for tomorrow
const H =range(0.15,stop = lbar, length = nH).^1 #Aggregate labor grid
K = range(5,stop = 20.0, length = nK).^1 #Aggregate capital grid

b, d,  nsim, asim, Ksim, Hsim,policygrid,K,R2b,R2d,zsim,esim = KrusselSmithENDOGENOUS(A,A1,E,Z,tmat,states,K,H,b,d;
N=N,T=T,discard = 1000, update_policy=0.9,updateb= 0.6, updaterule = true)

#@time b, d,  nsim, asim, Ksim, Hsim,policygrid,Vgrid,K,R2b,R2d,zsimd,esim =KrusselSmith(A,
#E,Z,tmat,states,K,H,b,d;tol= 1e-6,inner_optimizer = BFGS(linesearch =BackTracking()),
#discard = 1000,updateV = 0.7,updateb= 0.3,N=N,T=T)

using JLD2, FileIO
@save "save_variables/variables_nA$(nA).jld2" b d  nsim asim Ksim Hsim policygrid K R2b R2d zsim esim


#@load "variables.jld2"

#using Plots
#plot1 = plot(Ksim, label = ["Aggregate Capital" ], legend = :bottomright)
#savefig(plot1, "AggK.png")

#plot2 = plot(Hsim, label = ["Aggregate Labor"], legend = :bottomright)
#savefig(plot2, "AggH.png") #
