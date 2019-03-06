using Plots, NLsolve, ForwardDiff, DataFrames, LinearAlgebra, QuantEcon, Plots
using Optim, Statistics, NLSolversBase,LaTeXStrings

include("State_Space.jl")

#Parameters:
δ = 1   #depreciation rate
θ = 1/3  #capital share of output
β = 0.972  #Discouting
σ = 2  #Elasticity of Intertemporal Substitution
ψ = 3   #Labor parameter
γn= 0.01    #Population growth rate
γz= 0.03   #Productivitu growth rate


#Parameters to be estimated and here used in our simulated example
gss = log(0.01) #average g (in logs)
τxss = 0.01 #average τx
τhss = 0.01 #average τh
zss = log(1) #average z (in logs)

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

σg= 0.035
σx = 0.08
σz = 0.05
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
S = randn(5,T) #vector with shocks

#Simulating data
for t=1:T
    if t>1
    X[:,t] = A*X[:,t-1]+ B*S[:,t]
    end
    Y[:,t] = C*X[:,t]
end


plot([X[2,:],X[3,:],X[4,:],X[5,:]],title ="Wedges", labels = ["Z","tauh","taux","g"])
plot([X[1,:],Y[2,:],Y[1,:],Y[3,:]],title = "Endogenous Variables",labels = ["K","X","Y","L"])

#Only efficiency shocks
Xz= ones(5,T).* [0,0,0,0,0]
Yz = ones(4,T).*[0,0,0,0]
Sz = vcat(S[1:2,:], zeros(3,T))

for t=1:T
    if t>1
    Xz[:,t] = A*Xz[:,t-1]+ B*Sz[:,t]
    end
    Yz[:,t] = C*Xz[:,t]
end
plot([Yz[1,:],Y[1,:]],title = "Only efficiency wedge",labels = ["Yz","Y"])


#Only labor shocks
Xh= ones(5,T).* [0,0,0,0,0]
Yh = ones(4,T).*[0,0,0,0]
Sh = [zeros(2,T); S[3,:]' ; zeros(2,T)]


for t=1:T
    if t>1
    Xh[:,t] = A*Xh[:,t-1]+ B*Sh[:,t]
    end
    Yh[:,t] = C*Xh[:,t]
end
plot([Yh[1,:],Y[1,:]],title = "Only labor wedge",labels = ["Yh","Y"])

#Only investment shocks
Xx= ones(5,T).* [0,0,0,0,0]
Yx = ones(4,T).*[0,0,0,0]
Sx = [zeros(3,T); S[4,:]' ; zeros(1,T)]


for t=1:T
    if t>1
    Xx[:,t] = A*Xx[:,t-1]+ B*Sx[:,t]
    end
    Yx[:,t] = C*Xx[:,t]
end
plot([Yx[1,:],Y[1,:]],title = "Only investment wedge",labels = ["Yx","Y"])

#Only Government shocks
Xg= ones(5,T).* [0,0,0,0,0]
Yg = ones(4,T).*[0,0,0,0]
Sg = [zeros(4,T); S[5,:]']


for t=1:T
    if t>1
    Xg[:,t] = A*Xg[:,t-1]+ B*Sg[:,t]
    end
    Yg[:,t] = C*Xg[:,t]
end
plot([Yg[1,:],Y[1,:]],title = "Only government wedge",labels = ["Yg","Y"])

plot(plot([Yz[1,:],Y[1,:]],title = "Only efficiency wedge",labels = ["Yz","Y"]),
    plot([Yh[1,:],Y[1,:]],title = "Only labor wedge",labels = ["Yh","Y"]),
    plot([Yx[1,:],Y[1,:]],title = "Only investment wedge",labels = ["Yx","Y"]),
plot([Yg[1,:],Y[1,:]],title = "Only government wedge",labels = ["Yg","Y"]))

#No efficiency shocks
XZ= ones(5,T).* [0,0,0,0,0]
YZ = ones(4,T).*[0,0,0,0]
SZ = [zeros(2,T); S[3:5,:]]


for t=1:T
    if t>1
    XZ[:,t] = A*XZ[:,t-1]+ B*SZ[:,t]
    end
    YZ[:,t] = C*XZ[:,t]
end
plot([YZ[1,:],Y[1,:]],title = "No efficiency wedge",labels = ["YZ","Y"])

#No labor shocks
XH= ones(5,T).* [0,0,0,0,0]
YH = ones(4,T).*[0,0,0,0]
SH = [ S[1:2,:] ; zeros(1,T); S[4:5,:]]


for t=1:T
    if t>1
    XH[:,t] = A*XH[:,t-1]+ B*SH[:,t]
    end
    YH[:,t] = C*XH[:,t]
end
plot([YH[1,:],Y[1,:]],title = "No labor wedge",labels = ["YH","Y"])

#No investment shocks
XX= ones(5,T).* [0,0,0,0,0]
YX = ones(4,T).*[0,0,0,0]
SX = [ S[1:3,:] ; zeros(1,T); S[5,:]']


for t=1:T
    if t>1
    XX[:,t] = A*XX[:,t-1]+ B*SX[:,t]
    end
    YX[:,t] = C*XX[:,t]
end
plot([YX[1,:],Y[1,:]],title = "No investment wedge",labels = ["YX","Y"])

#No Government shocks
XG= ones(5,T).* [0,0,0,0,0]
YG = ones(4,T).*[0,0,0,0]
SG = [S[1:4,:];zeros(1,T)]


for t=1:T
    if t>1
    XG[:,t] = A*XG[:,t-1]+ B*SG[:,t]
    end
    YG[:,t] = C*XG[:,t]
end
plot([YG[1,:],Y[1,:]],title = "No government wedge",labels = ["YG","Y"])

plot(plot([YZ[1,:],Y[1,:]],title = "No efficiency wedge",labels = ["YZ","Y"]),
    plot([YH[1,:],Y[1,:]],title = "No labor wedge",labels = ["YH","Y"]),
    plot([YX[1,:],Y[1,:]],title = "No investment wedge",labels = ["YX","Y"]),
plot([YG[1,:],Y[1,:]],title = "No government wedge",labels = ["YG","Y"]))

#If real data is wanted
#DATA = loaddata()
#Y = vcat(DATA[:GDP_dev]',DATA[:Investment_dev]',DATA[:Labor_dev]',DATA[:GOV_dev]')
