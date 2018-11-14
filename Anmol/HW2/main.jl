using Plots, NLsolve, ForwardDiff, DataFrames, LinearAlgebra, QuantEcon, Plots
using Optim, Statistics, NLSolversBase,LaTeXStrings
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW2")
include("State_Space.jl")
include("load_data.jl")
include("KalmanFilter.jl")
#Parameters:
δ = 0.0464   #depreciation rate
θ = 1/3  #capital share of output
β = 0.972  #Discouting
σ = 1  #Elasticity of Intertemporal Substitution
ψ = 3    #Labor parameter
γn= 0.00    #Population growth rate
γz= 0.00   #Productivitu growth rate


#Parameters to be estimated and here used in our simulated example
gss = 0.0 #average g
τxss = 0.0 #average τx
τhss = 0.0 #average τh
zss = 0.0 #average z (z is in logs)

#Parameters to be estimated
ρg = 0.85
ρx = 0.73
ρh = 0.74
ρz = 0.98
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
σzg= 0.02
σzx = 0.001
σzh = 0.02
σhx = 0.01
σhg = 0.02
σxg = 0.02

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

T=500
X= zeros(5,T)
Y = zeros(4,T)

#Simulating data
for t=1:T
    if t>1
    X[:,t] = A*X[:,t-1]+ B*randn(5,1)
    end
    Y[:,t] = C*X[:,t]
end



#If real data is wanted
#DATA = loaddata()
#Y = vcat(DATA[:GDP_dev]',DATA[:Investment_dev]',DATA[:Labor_dev]',DATA[:GOV_dev]')
