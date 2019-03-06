
using NLsolve, ForwardDiff, LinearAlgebra, Random, JLD2,FileIO
using Optim, Statistics, NLSolversBase,LaTeXStrings
include("State_Space.jl")
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
gss = log(0.01) #average g (in logs)
τxss = 0.05 #average τx
τhss = 0.02 #average τh
zss = log(1) #average z (z is in logs)

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
@time A,B,C = State_Space(params_calibrated,steadystates, P,Q)


T=5000
X= zeros(5,T)
Y = zeros(4,T)

Random.seed!(0403);
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


original = [ρg,ρx,ρh,ρz,ρzg,ρzx,ρzh,ρhz,ρhx,ρhg,ρxz,ρxh,ρxg,ρgz,ρgx,ρgh,σg,σx,σz,σh,σzg,σzx,σzh,σhx,σhg,σxg,gss,τxss,τhss,zss]
#Initial guess
truelikelihood = maxloglikelihood(original)

Random.seed!(0403);
initial = original .+ randn(length(original))*0.1



#Solver Stuff
inner_optimizer = LBFGS() #LBFGS()  # SimulatedAnneaestimates - originalling() #NelderMead() ConjugateGradient()
println("Starting Likelihood maximization")
println("Solver is: $(inner_optimizer)"[1:25])


#Defining lower and upper bounds for estimator
lower=zeros(length(initial))

upper = ones(length(initial))
if length(initial) == 8
    upper[5:8] = 0.05 *ones(4)
end
if length(initial)>16
    upper[17:26] = 0.05.*ones(10)
    lower[5:16] = -1*ones(12)
end
if length(initial) == 30
    upper[27:30] = 0.1 * ones(4)
    lower[30] = -100 #z lower bound (in logs)
    lower[27] = -100 #g lower bound (in logs)
end
#making sure that the initial guess is within the bounds
initial = min.(upper.*0.99,initial)
initial = max.(lower.+0.0001,initial)

maxloglikelihood(initial)


bla = optimize(maxloglikelihood,lower,upper, initial,Fminbox(inner_optimizer),Optim.Options(show_trace = true, show_every = 5,iterations =500, time_limit = 60*60*1.0))

bla.minimum - truelikelihood

estimates = bla.minimizer

@save "results_$(length(original))_$("$(inner_optimizer)"[1:5]).jld2"
