#Author: Joao Lazzaro
#This script appoximates a discouted utility fnction via 2nd order Taylo Expansion
#And solves the Riccati equations using the Vaughan method.
#We plot the capital time series in the end.
#We need to find matrices Q,R, W such that
#U(X,u)≈X′QX+u′Ru+2u′WX=[X u]′M[X u]
#Where X is a vector of states variables and u of cotrol variables
include("Vaughan.jl")


using Plots
using ForwardDiff
using NLsolve


#Parameters:
δ = 1    #depreciation rate
θ = 0.25 #capital share of output
β = 0.9  #Discouting
ρ = 0.5  #AR coefficient
σ = 0  #AR shock SD
μ = (1/(1-θ)) * log((1-β*(1-δ))/(θ * β))    #This transformations are needed to set steady state value of capital to 1
ϕ = 0    #Labor parameter
γn= 0    #Population growth rate
γz= 0    #Productivitu growth rate

Xnneg = true #Non negative investment


#return function is the utility function:
#Euler Equation:
#using NLsolve we need to defie the Euler equation as functions
function ee!(eq, x)
    h=(x[2])
    k=(x[1])
    eq[1] = k - (β*θ / ((1+γz)-β*(1-δ)))^(1/(1-θ)) *h*exp(μ)
    eq[2] = (1-h)*((1-θ) *k^θ *exp(μ)^(1-θ) *h^(-θ)) - ϕ*(k^θ * (exp(μ)*h)^(1-θ) +(1-δ)*k - (1+γn)*(1+γz)k)
end
S = nlsolve(ee!, [0.1,0.8],ftol = :1.0e-9, method = :trust_region , autoscale = true)
#kss=exp(μ)*((1-β*(1-δ))/(θ*β))^(1/(θ-1))
kss = S.zero[1] #get the SS values
hss = S.zero[2]
zss = μ #SS of the shocks is just 0 ote that z goes into the utility funcion as exp(z)

#Utility function
#REMEMBER: V is the [X u]' Vector, hence [k0 z0 1 k1 h0]'
#X is the state variables vector (and a constant) while u is the cotrol variables vector
function u(x::Vector)
    c = x[1]^θ *(exp(x[2])*x[5])^(1-θ) +(1-δ)*x[1] - (1+γn)*(1+γz)*x[4]
    investment =  (1+γn)*(1+γz)*x[4] - (1-δ)*x[1]
    if c<=0
        u = -Inf #any value of negative consumtpion in the grid will be avoided
    elseif investment <0
        u=-Inf #nonnegative investment constraint

    elseif ϕ >0
        u = log(c) + ϕ * log(1-x[5])
    else
        u=log(c)
    end
    return u
end

#vector with SS variables And the consant term
vss= [kss,zss,1,kss,hss]
#Find the Gradient and Hessian AT THE SS
uss=u(vss)
∇u = ForwardDiff.gradient(u,vss)
Hu = ForwardDiff.hessian(u,vss)

Xss = vss[1:3]
Uss = vss[4:5]

#We are looking for a matrix M such that u(vss) = vss'M vss
#Applying Sargent's Formula:
e = [0;0;1;0;0] #thus e'v=1


M = e.* (uss - ∇u'*vss .+ (0.5 .* vss' *Hu *vss) ) * e' +
    0.5 *(∇u * e' - e * vss' * Hu - Hu*vss*e' +e* ∇u') +
    0.5 * Hu

#Translating M into the Matrices we need:
Q = M[1:3,1:3]
W = M[1:3,4:5]
R = M[4:5,4:5]

A = [0 0 0 ;0 ρ 0 ; 0 0 1]
B = [1  0; 0  0 ; 0 0]
C = [0 ; 1 ; 0]

#Mapping to the problem without discounting (1 VARIABLES ARE tilde IN LECTURE NOTES)
A1 = sqrt(β) *(A -B*inv(R)*W')
B1 = sqrt(β) *B
Q1 = Q- W*inv(R)*W'


Pv,Fv = Vaughan(A1,B1,Q1,R,W)
F1v= Fv-inv(R) *W'

T= 10 #simulation time
ϵ = σ .* randn(T) #Normal distributed shocks
z =zeros(T) #initial state is 0 - the ss
k = ((log(1/2)) *randn())*ones(T) #initial state is not the ss

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

k_vaughan = U[1,:]
h = U[2,:]
z = X[2,:]

labels = "Capital"
plot(k_vaughan,label = labels)
