include("functions.jl")

#= δ  depreciation rate
θ   #capital share of output
β   #Discouting
σ   #Elasticity of Intertemporal Substitution
ψ     #Labor parameter
γn    #Population growth rate
γz   #Productivitu growth rate
gss  #average g
τxss  #average τx
τhss  #average τh
zss  #average z (z is in logs) =#


#Parameters:
δ = 1#0.0464   #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
σ = 1 #Elasticity of Intertemporal Substitution
ψ = 1   #Labor parameter
γn= 0.00    #Population growth rate
γz= 0.00   #Productivitu growth rate


#Parameters to be estimated and here used in our simulated example
gss = 0.03 #average g (in logs)
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

σg= 0.01
σx = 0.01
σz = 0.01
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


R, B1y,P,B2,A2y,kss,Wy =   LQ_distorted(params_calibrated,steadystates)

K = range(1/5*kss,stop=3*kss,length =300)


policy = -(R+B1y'*P*B2)\B1y'*P*A2y

u1= zeros(300,2)
uf = copy(u1)

for i=1:300
    global u,u1
    u1[i,:] += policy*[K[i],zss,τhss, τxss, gss,1]
    uf[i,:] = u1[i,:]/(sqrt(β)) - (R\Wy')*[K[i],zss,τhss, τxss, gss,1]
end

using Plots
plot(K,[K uf[:,1]],labels = ["45" "LQ"])
