#Author: Joao Lazzaro
#This script appoximates a discouted utility fnction via 2nd order Taylo Expansion
#And solves the Riccati equations using the Vaughan method.
#We plot the capital time series in the end.
#We need to find matrices Q,R, W such that
#U(X,u)≈X′QX+u′Ru+2u′WX=[X u]′M[X u]
#Where X is a vector of states variables and u of cotrol variables
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Ellen\\HW2")
include("Vaughan.jl")


using Plots
using ForwardDiff
using NLsolve


#Parameters:
δ = 1    #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
ρ = 0.5  #AR coefficient
σ = 0.5  #AR shock SD
μ = 0    #AR(1) constant term
ϕ = 0    #Labor parameter
γn= 0    #Population growth rate
γz= 0    #Productivitu growth rate

Xnneg = true #Non negative investment



T= 10000 #simulation time
ϵ = σ .* randn(T) #Normal distributed shocks
z =zeros(T) #initial state is 0 - the ss
k = (kss+σ *randn())*ones(T) #initial state is not the ss

X = [k';z';ones(T)']
X1= copy(X)
U = ones(2,T)
U[:,1] = -F1v*X[:,1]
U1=copy(U)
for t=1:T-1
    X[:,t+1] = ((A-B*Fv)*X[:,t] + C.* ϵ[t+1])
    X1[:,t+1]= β^(1/2)*X[:,t+1]
    U[:,t+1] = -Fv* X[:,t+1]
    U1[:,t+1] = -F1v* X1[:,t+1]
end

k = U[1,:]
h = U[2,:]
z = X[2,:]

labels = "Capital"
plot(k,label = labels)
