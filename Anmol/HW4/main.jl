include("functions.jl")
include("Carrol_KS.jl")
#Defining parameters they were taken frmom KS
const α = 0.36 #capital share in output
const β = 0.99 #Impatience
const δ = 0.025 #Depreciation rate

const η = 1.0/2.9 #labor /consumption elasticity
const μ = 1.0 #Intratemporal elasticity

const N = 5000 #number of agents in each economy
const T = 11000 #number of simulation periods
const discard = 1000 #number of discarded periods
const lbar = 1.0#/0.9 #time endowment

const nA = 30 #number of assets gridpoints
const nK = 6 #Number of aggregate capital gridpoints
const nH = 6   #Number of aggregate labor gridpoints

amin = 0.0
amax = 350.0

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
d = [-0.592 -0.255;-0.544 -0.252]#[log(0.304) -1.3;log(0.304) -1.28]#
#Asset grid:
factor =4.0
const A =(range(0, stop=nA-1, length=nA)/(nA-1)).^factor * amax #Capital grid for todayconst A1 = A#(range(0, stop=nA-1, length=nA)/(nA-1)).^factor * amax #Capital grid for today#range(0., stop=amax, length=nA).^1 #Capital grid for tomorrow
const A1 = A
K = range(5,stop = 20.0, length = nK).^1 #Aggregate capital grid

Hk= ones(nK,2) .*lbar
for j=1:2,i =1:nK-1
    Hk[i,j] = H0(K[i],Z[j])
end

#H = (range((nH-1)/(10*maximum(Hk)), stop=nH-1, length=nH)/(nH)).^1.0 * (maximum(Hk))

H = range(0.001,stop = lbar,length = nH).^1.0

b, d,  nsim, asim, Ksim, Hsim,policygrid,K,R2b,R2d,zsim,esim = KrusselSmithENDOGENOUS(A,A1,E,Z,tmat,states,K,H,b,d;
N=N,T=T,discard = discard, update_policy=.75,updateb= .4, updaterule = true,seed =1234)



using JLD2, FileIO
@save "save_variables/variables_nA$(nA).jld2" b d  nsim asim Ksim Hsim policygrid K R2b R2d zsim esim discard Z T


#@load "variables.jld2"

using Plots
#Aggregate Capital vs capital:
plot((Ksim[discard+1:end-1][zsim[1:end-1].==Z[1]]) ,(Ksim[discard+2:end][zsim[1:end-1].==Z[1]]),
xlabel = "\$K_t\$",ylabel = "\$K_{t+1}\$",linestyle = :dot, label = "Bad State",legend = :bottomright)
plot!((Ksim[discard+1:end-1][zsim[1:end-1].==Z[2]]) ,(Ksim[discard+2:end][zsim[1:end-1].==Z[2]]),
linestyle = :dot, label = "Good State")
plot!(11.7:0.1:12.7 ,11.7:0.1:12.7, label = "45 degrees")
#Aggregate labor vs capital:
plot((Ksim[discard+1:end][zsim.==Z[1]]) ,(Hsim[discard+1:end][zsim.==Z[1]]),
xlabel = "\$K_t\$",ylabel = "\$H_t\$",linestyle = :dot, label = "Bad State")
plot!((Ksim[discard+1:end][zsim.==Z[2]]) ,(Hsim[discard+1:end][zsim.==Z[2]]),
linestyle = :dot,label = "Good State")
