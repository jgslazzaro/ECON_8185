using Plots, NLsolve, ForwardDiff, DataFrames, LinearAlgebra, QuantEcon, Plots, Optim, Statistics
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW2")
include("State_Space.jl")
include("load_data.jl")
include("KalmanFilter.jl")
#Parameters:
δ = 1   #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
σ = 2  #Elasticity of Intertemporal Substitution
ψ = 1    #Labor parameter
γn= 0.00    #Population growth rate
γz= 0.00   #Productivitu growth rate
gss = 0.01 #average g
τxss = 0.02 #average τx
τhss = 0.02 #average τh
zss = 0.0 #average z (z is in logs)

#Parameters to be estimated
ρg = 0.7
ρx = 0.7
ρh = 0.7
ρz = 0.7

σg= 0.01
σx = 0.01
σz = 0.01
σh = 0.01

#In matrix form
P = [ρz  0 0 0;
0 ρh 0  0;
0 0 ρx 0;
0 0 0 ρg]
Q = [σz 0 0 0;
0 σh 0  0;
0 0 σx 0;
0 0 0 σg]

params_calibrated = [δ,θ,β,σ,ψ,γn,γz,gss,τxss,τhss,zss]

A,B,C = State_Space(params_calibrated, P,Q)

T=300
X= zeros(5,T)
Y = zeros(4,T)


for t=1:T

    if t>1
    X[:,t] = A*X[:,t-1]+ B*randn(5,1)
    end
    Y[:,t] = C*X[:,t]
end


DATA = loaddata()

#Y = vcat(DATA[:GDP_dev]',DATA[:Investment_dev]',DATA[:Labor_dev]',DATA[:GOV_dev]')
A,B,C = State_Space(params_calibrated, P,Q)
X, a, Ω = KalmanFilter(Y,A,B,C)
likelihood(Y,Ω,a)


#plot([X[5,:], Y[4,:]], labels = ["Estimated","Actual"])
#plot([X[1,:],X[2,:],Y[2,:],Y[1,:],Y[3,:]],labels = ["K","Z","X","Y","L"])

initial = [0.5,0.5,0.5,0.5,0.02,0.02,0.02,0.02]
d=10
lower = zeros(8)
upper = [1.0,1,1,1,1,1,1,1]
while d>10^(-7)
    global d, initial
    bla = optimize(maxloglikelihood,lower, upper, initial, Fminbox(GradientDescent()))
    d = maximum(abs.(initial - bla.minimizer))
    println(d)
    initial = bla.minimizer
end

ρz,ρh,ρx,ρg,σz,σh,σx,σg = initial
A,B,C = State_Space(params_calibrated, P,Q)
#Parameters to be estimated
ρg = 0.7
ρx = 0.7
ρh = 0.7
ρz = 0.7

σg= 0.01
σx = 0.01
σz = 0.01
σh = 0.01
