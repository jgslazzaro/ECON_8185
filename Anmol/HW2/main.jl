using Plots, NLsolve, ForwardDiff, DataFrames, LinearAlgebra, QuantEcon
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW2")
include("State_Space.jl")
include("load_data.jl")
#Parameters:
δ = 1   #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
σ = 2  #Elasticity of Intertemporal Substitution
ψ = 1    #Labor parameter
γn= 0.00    #Population growth rate
γz= 0.00   #Productivitu growth rate
gss = 0.0 #average g
τxss = 0.0 #average τx
τhss = 0.0 #average τh
zss = 0.0 #average z (z is in logs)

#Parameters to be estimated
ρg = 0.0
ρx = 0.0
ρh = 0.0
ρz = 0.8

σg= 0.0
σx = 0.0
σz = 0.08
σh = 0.0

#In matrix form
P = [ρz 0 0 0;
0 ρh 0 0 ;
0 0 ρx 0 ;
0 0 0 ρg]
Q = [σz 0 0 0;
0 σh 0 0 ;
0 0 σx 0 ;
0 0 0 σg]

params_calibrated = [δ,θ,β,σ,ψ,γn,γz,gss,τxss,τhss,zss]

A,B,C = State_Space(params_calibrated, P,Q)


#defining the vectors
T=100
X= ones(5,T).* [0,0,0,0,0]
Y = ones(3,T).*[0,0,0]


for t=1:T

    if t>1
    X[:,t] = A*X[:,t-1]+ B*randn(5,1)
    end
    Y[:,t] = C*X[:,t]
end
t=2
A*X[:,t-1]
plot([X[1,:],X[2,:],Y[2,:],Y[1,:],Y[3,:]],labels = ["K","Z","X","Y","L"])
