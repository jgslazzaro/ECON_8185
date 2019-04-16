
#Author: Joao Lazzaro
#This script appoximates a discouted utility fnction via 2nd order Taylo Expansion
#And solves the Riccati equations. We plot the capital time series in the end.
#We need to find matrices Q,R, W such that
#U(X,u)≈X′QX+u′Ru+2u′WX=[X u]′M[X u]
#Where X is a vector of states variables and u of cotrol variables
#]("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Ellen\\HW2")
include("Riccati.jl")
using ForwardDiff
using NLsolve


#Parameters:
δ = 0.1    #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
ρ = 0.5  #AR coefficient
σ = 0.5  #AR shock SD
μ = 0    #AR(1) constant term
ϕ = 2    #Labor parameter
γn= 0    #Population growth rate
γz= 0    #Productivitu growth rate

Xnneg = true #Non negative investment


A,B,C,P ,F, F1,kss,hss = LQ(δ,β,ρ,σ,μ,ϕ,γn,γz)



T= 10000 #simulation time
ϵ = σ .* randn(T) #Normal distributed shocks
z =zeros(T) #initial state is 0 - the ss
k = (kss+σ *randn())*ones(T) #initial state is not the ss

X = [k';z';ones(T)']
X1= copy(X)
U = ones(2,T)
U[:,1] = -F1*X[:,1]
U1=copy(U)
for t=1:T-1
    X[:,t+1] = ((A-B*F)*X[:,t] + C.* ϵ[t+1])
    X1[:,t+1]= β^(1/2)*X[:,t+1]
    U[:,t+1] = -F* X[:,t+1]
    U1[:,t+1] = -F1* X1[:,t+1]
end

k = X[1,:]
h = U[2,:]
z = X[2,:]

#Plots:
using Plots
labels = "Capital"
plot(k,label = labels)

labels = "Labor"
plot(h,label = labels)
