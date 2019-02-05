using Plots, NLsolve, ForwardDiff, DataFrames, LinearAlgebra, QuantEcon, Plots
using Optim, Statistics, NLSolversBase,LaTeXStrings
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW2")
include("State_Space.jl")
include("load_data.jl")
include("KalmanFilter.jl")
#Parameters:
δ = 0.0464   #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
σ = 1  #Elasticity of Intertemporal Substitution
ψ = 1    #Labor parameter
γn= 0.00    #Population growth rate
γz= 0.00   #Productivitu growth rate


#Parameters to be estimated and here used in our simulated example
gss = 0.1 #average g
τxss = 0.05 #average τx
τhss = 0.02 #average τh
zss = 0.0 #average z (z is in logs)

#Parameters to be estimated
ρg = 0.8
ρx = 0.5
ρh = 0.7
ρz = 0.9
ρzg= 0.0
ρzx = 0.0
ρzh = 0.0
ρhz = 0.0
ρhx = 0.0
ρhg = 0.0
ρxz = 0.0
ρxh = 0.0
ρxg = 0.0
ρgz = 0.0
ρgx = 0.0
ρgh = 0.0

σg= 0.02
σx = 0.01
σz = 0.02
σh = 0.01
σzg= 0.0
σzx = 0.00
σzh = 0.0
σhx = 0.0
σhg = 0.0
σxg = 0.0

#In matrix form
P = [ρz ρzh ρzx ρzg;
ρhz ρh ρhx ρhg ;
ρxz ρxh ρx ρxg ;
ρgz ρgh ρgx ρg]
Q = [σz σzh σzx σzg;
σzh σh σhx σhg ;
σzx σhx σx σxg ;
σzg σhg σxg σg]

params_calibrated = [δ,θ,β,σ,ψ,γn,γz,]
steadystates = [gss,τxss,τhss,zss]
A,B,C = State_Space(params_calibrated,steadystates, P,Q)

T=300
X= zeros(5,T)
Y = zeros(4,T)
S = randn(5,T) #vector with shocks

#Simulating data
for t=1:T
    if t>1
    X[:,t] = A*X[:,t-1]+ B*S[:,t]
    end
    Y[:,t] = C*X[:,t]
end


#plot([X[2,:],X[3,:],X[4,:],X[5,:]],title ="Wedges", labels = ["Z","tauh","taux","g"])
#plot([X[1,:],Y[2,:],Y[1,:],Y[3,:]],title = "Endogenous Variables",labels = ["K","X","Y","L"])



#Initial guess
initial = [0.795,0.715,0.495,0.5,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001] #ρz,ρh,ρx,ρg,σz,σh,σx,σg,gss,τxss,τhss,zss
d=10
lower = zeros(12) #Lower bound for the parameters
upper = [1.0,1.0,1.0,1.0,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05] #Upper bound.
#This was kind of random
#Canova recommend to run optimization until the estimates converge
#This loop here does this
while d>10^(-4)
    global d, initial
    bla = optimize(maxloglikelihood,lower, upper, initial)
    d = maximum(abs.(initial - bla.minimizer))
    println(d)
    initial = bla.minimizer
end
#Get the results
ρz,ρh,ρx,ρg,σz,σh,σx,σg,gss,τxss,τhss,zss = initial
Pe = [ρz  0 0 0;
0 ρh 0  0;
0 0 ρx 0;
0 0 0 ρg]

Qe = [σz 0 0 0;
0 σh 0  0;
0 0 σx 0;
0 0 0 σg]
