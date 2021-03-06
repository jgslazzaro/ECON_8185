#Parameters:
δ = 0.0464   #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
σ = 1 #Elasticity of Intertemporal Substitution
ψ =  1  #Labor parameter
γn= 0.00    #Population growth rate
γz= 0.00   #Productivitu growth rate
β=β*(1+γn)

#Parameters to be estimated and here used in our simulated example
gss = 0.01 #average g (in logs)
τxss = 0.01 #average τx
τhss = 0.02 #average τh
zss = log(1) #average z (z is in logs)

#Parameters to be estimated
ρg = 0.2
ρx = 0.4
ρh = 0.6
ρz = 0.8
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

σg= 0.05
σx = 0.02
σz = 0.1
σh = 0.07
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
include("functions.jl")

#First get the LQ policy function:

R, B1y,P,F1,B2,A2y,kss,hss,Wy =   LQ_distorted(params_calibrated,steadystates)

K = range(kss-2,stop=kss+2,length =300)

u1= zeros(300,2)
uf = copy(u1)



for i=1:300
    global u,u1
    u1[i,:] += -F1*[K[i],zss,τhss, τxss, gss,1]
    uf[i,:] = u1[i,:]/(sqrt(β)) - (R\Wy')*[K[i],zss,τhss, τxss, gss,1]
end


using Plots
plot(K,[K uf[:,1]],labels = ["45" "LQ"],legend = :bottomright, title = "LQ capital policy function")
savefig("figure1.png")


#Let's now work with the Vaughan method
#In matrix form
P = [ρz ρzh ρzx ρzg;
ρhz ρh ρhx ρhg ;
ρxz ρxh ρx ρxg ;
ρgz ρgh ρgx ρg]
Q = [σz σzh σzx σzg;
σzh σh σhx σhg ;
σzx σhx σx σxg ;
σzg σhg σxg σg]

A,B,C = State_Space(params_calibrated,steadystates, P,Q)

T=200

X= zeros(5,T)
Y = zeros(4,T)
using Random
Random.seed!(1234)
S = randn(5,T) #vector with shocks

#Simulating data
for t=1:T
    if t>1
    X[:,t] = A*X[:,t-1]+ B*S[:,t]
    end
    Y[:,t] = C*X[:,t]
end

#Plots
plot([X[2,:],X[3,:],X[4,:],X[5,:]],title ="Wedges", labels = ["Z","tau_h","taux","g"],legend=:bottomright)
savefig("figure2.png")
plot([X[1,:],Y[2,:],Y[1,:],Y[3,:]],title = "Endogenous Variables",labels = ["K","X","Y","L"])
savefig("figure3.png")

w =  θ*(X[1,:].-Y[3,:]) .+ (1-θ)*X[2,:]
dividends = Y[1,:].- Y[3,:] .-Y[2,:].+w



plot(dividends,labels = ["Dividends"],legend = :bottomright)
savefig("figure4.png")


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

savefig("figure5.png")
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
savefig("figure6.png")
