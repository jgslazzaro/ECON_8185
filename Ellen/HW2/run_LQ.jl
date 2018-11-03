
#Author: Joao Lazzaro
#This script appoximates a discouted utility fnction via 2nd order Taylo Expansion
#And solves the Riccati equations. We plot the capital time series in the end.
#We need to find matrices Q,R, W such that
#U(X,u)≈X′QX+u′Ru+2u′WX=[X u]′M[X u]
#Where X is a vector of states variables and u of cotrol variables
include("Vaughan.jl")
using Plots
using ForwardDiff
using NLsolve


#Parameters:
δ = 1    #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
ρz = 0.5  #AR coefficient
ρg = 0.0
ρx = 0.0
ρh = 0.0
σz = 0.05  #AR shock SD
σx = 0.0
σh = 0.0
σg = 0.0
μ = 0    #AR(1) constant term
ϕ = 0    #Labor parameter
γn= 0    #Population growth rate
γz= 0    #Productivitu growth rate
N0= 1   #Initial Population

τhss =  0.05
τxss =  0.05
gss = 0.05


Xnneg = true #Non negative investment


#return function is the utility function:
#Euler Equation:
#using NLsolve we need to defie the FOCs as functions
function ss!(eq, x)
    h= x[2]
    k= x[1]

    #Market Clearing:
    K = N0*x[1]
    L = N0*x[2]

    #Firm FOC
    r = θ*K^(θ-1)*L^(1-θ)
    w = (1-θ)*K^(θ)*L^(-θ)
    #Government transfer is the difference between revenues and g
    T = τhss*w*L +τxss*((1+γn)*(1+γz)*K-(1-δ)*K) - gss
    c = r*k+(1-τhss)w*h+ T  - (1+τxss)*((1+γn)*(1+γz)*k-(1-δ)*k)
    #Agent FOC:
    eq[1] = (1+γn)*(1+γz)*(1+τxss) - β*(1+γn) *( r + (1+τxss)*(1-δ)*k)
    eq[2] = (1-h)*(1-τhss)*w - ϕ*c

end
S = nlsolve(ss!, [0.05,0.9],ftol = :1.0e-9, method = :trust_region , autoscale = true)

#kss=exp(μ)*((1-β*(1-δ))/(θ*β))^(1/(θ-1))
kss = S.zero[1] #get the SS values
hss = S.zero[2]
Kss = N0*kss
Lss = N0*hss

zss = 0 #SS of the shocks is just 0 ote that z goes into the utility funcion as exp(z)

#Utility function
#REMEMBER: V is the [X u]' Vector, hence [1 k0 z0 g τx τh K L k1 h0]'
#X is the state variables vector (and a constant) while u is the cotrol variables vector
function u(x::Vector)
    k0 = x[2]
    z = x[3]
    g = x[4]
    τx = x[5]
    τh = x[6]
    K = x[7]
    L = x[8]
    k1 = x[9]
    h = x[10]

    #Firm FOC
    r = θ*K^(θ-1)*L^(1-θ)
    w = (1-θ)*K^(θ)*L^(-θ)
    #Government transfer is the difference between revenues and g
    T = 1/N0 * (τh*w*L +τx*((1+γn)*(1+γz)*K-(1-δ)*K) - g)
    c = r*k0+(1-τh)w*h+ T  - (1+τx)*((1+γn)*(1+γz)*k1-(1-δ)*k0)

    investment =  (1+γn)*(1+γz)*k0 - (1-δ)*k1
    if c<=0
        u = -Inf #any value of negative consumtpion in the grid will be avoided
    elseif investment <0
        u=-Inf #nonnegative investment constraint

    elseif ϕ >0
        u = log(c) + ϕ * log(1-h)
    else
        u=log(c)
    end
    return u
end

#vector with SS variables And the consant term
vss= [1,kss,zss,gss, τxss,τhss, Kss,Lss,kss,hss]
#Find the Gradient and Hessian AT THE SS
uss = u(vss)
∇u = ForwardDiff.gradient(u,vss)
Hu = ForwardDiff.hessian(u,vss)

Xss = vss[1:8]
Uss = vss[9:10]

#We are looking for a matrix M such that u(vss) = vss'M vss
#Applying Sargent's Formula:
e = [1;0;0;0;0;0;0;0;0;0] #thus e'v=1


M = e.* (uss - ∇u'*vss .+ (0.5 .* vss' *Hu *vss) ) * e' +
    0.5 *(∇u * e' - e * vss' * Hu - Hu*vss*e' +e* ∇u') +
    0.5 * Hu
#Translating M into the Matrices we need:
Q = M[1:8,1:8]
W = M[1:8,9:10]
R = M[9:10,9:10]

Wy = W[1:6,:]
Wz= W[7:8,:]

A = [1 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0;
0 0 ρz 0 0 0 0 0;
0 0 0 ρg 0 0 0 0;
0 0 0 0 ρx 0 0 0;
0 0 0 0 0 ρh 0 0;
0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0]

Ay=A[1:6,1:6]
Az = A[1:6,7:8]
B = [0 0;
1 0;
0 0;
0 0;
0 0;
0 0;
0 0;
0 0]



Wy=W[1:6,:]

C = [0 ;
 σz ;
 σg;
 σx;
 σh;
 0;
 0;
 0]

#Mapping to the problem without discounting (1 VARIABLES ARE ~ IN LECTURE NOTES)
A1 = sqrt(β) *(A -B*inv(R)*W')
B1 = sqrt(β) *B
Q1 = Q- W*inv(R)*W'


Ay1 = sqrt(β) *(Ay -By*inv(R)*Wy')
By1 = sqrt(β) *By
Az1 = sqrt(β) *(Az -By*inv(R)*Wz')
Q1 = Q- W*inv(R)*W'
Qy1 = Q1[1:6,1:6]
Qz1= Q1[1:6,7:8]


Θ = [0 0 0 0 0 0; 0 0 0 0 0 0]
Φ = [1 0; 0 1]

Θ1 = inv(I + Φ*inv(R)*Wz')*(Θ-Φ*inv(R)*Wy')
Φ1 = inv(I + Φ*inv(R)*Wz')*Φ

Atilde = Ay1 - By1*inv(R)Φ1'*Qz1'
Ahat = Ay1 +Az1 *Θ1
Qhat = Qy1 + Qz1*Θ1
Bhat = By1 +Az1*Φ1

P, F = Vaughan_dist(Atilde,Ahat,Bhat,By1,Qhat,R,W)

F1= F-inv(R) *W'

σ = [0;σz;σg;σx;σh;0;0;0]

T= 1000  #simulation time
#ϵ = σ .* randn(8) #Normal distributed shocks
z =zeros(T) #initial state is 0 - the ss
g = gss.+zeros(T)
τx = τxss .+zeros(T)
τh = τhss .+zeros(T)
k = (kss+σ[1] *randn())*ones(T) #initial state is not the ss

X = [k';z'; g';τx';τh';k'; ones(T)'; ones(T)']
X1= copy(X)
U = ones(2,T)
U[:,1] = -F1*X[:,1]
U1=copy(U)
for t=1:T-1
    X[:,t+1] = (A-B*F)*X[:,t] + Diagonal(C)* (σ .* randn(8))
    X1[:,t+1]= β^(1/2)*X[:,t+1]
    U[:,t+1] = -F* X[:,t+1]
    U1[:,t+1] = -F1* X1[:,t+1]
end

k = X1[1,:]
h = U[2,:]
z = X[2,:]

#Plots:
labels = "Capital"
plot(k,label = labels)

#plot(h,label = labels)
