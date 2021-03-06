#Author: Joao Lazzaro
#This code defines a fuction to iterate the Riccati equation as in Ellen's notes
using ForwardDiff
using NLsolve
using LinearAlgebra
function Riccati(A,B,Q,R,W,β,γ1=10^(-5),γ2=10^(-5),p=1)

#γ s are the convergece parameter ad p is the matrix p norm we are usig
#Guess initial value for P



P=ones(size(A)[2],size(A)[1])
F=ones(size(R)[1],size(W)[1])

#initialize loop:
distP = 10; distF = 10; i=1

#see the notes to understand this. It is basically a translation from there to Julia
while distP>=γ1.*opnorm(P,p) || distF>=γ2*opnorm(F,p)
    #global P, F, i, distP, distF, P1, F1
    P1 = Q + A'*P*A - A'*P*B*((R+B'* P *B)\B')*P*A
    F1 = (R+B'*P*B) \ B' * P * A
    distP = opnorm(P1-P,p)
    distF = opnorm(F1-F,p)
    i=i+1

    P=copy(P1)
    F = copy(F1)
end

F = F+R\W'

return P, F
end


#Utility function
#REMEMBER: V is the [X u]' Vector, hence [k0 z0 1 k1 h0]'
#X is the state variables vector (and a constant) while u is the cotrol variables vector
function u(x::Vector;β = β,δ = δ,θ = θ,ϕ = ϕ,γn=γn,γz = γz)
    c = x[1]^θ *(exp(x[2])*x[5])^(1-θ) -(1+γz) *(1+γn)x[4]  + (1-δ)*x[1]
    investment = (1+γz) *(1+γn)x[4]  - (1-δ)*x[1]
    if c<=0
        u = -Inf #any value of negative consumtpion in the grid will be avoided
    #elseif investment <0
    #    u=-Inf #nonnegative investment constraint

    elseif ϕ >0
        u = log(c) + ϕ * log(1-x[5])
    else
        u = log(c)
    end
    return u
end


#Euler Equation:
#using NLsolve we need to define the Euler equation as functions
function ee!(eq, x;β = β,δ = δ,θ = θ,ϕ = ϕ,γn=γn,γz = γz)
    h=(x[2])
    k=(x[1])
    eq[1] = k^(1-θ) - (β*θ / ((1+γz)-β*(1-δ))) *h^((1-θ))
    eq[2] = (1-h)*((1-θ) *k^θ * h^(-θ)) - ϕ*(k^θ * h^(1-θ) +(1-δ)*k - (1+γn)*(1+γz)k)
end

function LQ(δ,β,ρ,σ,μ,ϕ,γn,γz)

    #Getting the Steady States from Euler Equations
    S = nlsolve(ee!, [0.1,0.8],ftol = :1.0e-9, method = :trust_region , autoscale = true)
    #kss=exp(μ)*((1-β*(1-δ))/(θ*β))^(1/(θ-1))
    kss = S.zero[1] #get the SS values
    hss = S.zero[2]
    zss = 0.0 #SS of the shocks is just 0 ote that z goes into the utility funcion as exp(z)


    #vector with SS variables And the consant term
    vss= [kss,zss,1.0,kss,hss]
    #Find the Gradient and Hessian AT THE SS
    uss=u(vss)
    ∇u = ForwardDiff.gradient(u,vss)
    Hu = ForwardDiff.hessian(u,vss)

    Xss = vss[1:3]
    Uss = vss[4:5]

    #We are looking for a Linear quadratic matrix M such that u(vss) = vss'M vss
    #Applying Sargent's Formula for 2nd order Taylor Expansion:
    e = [0;0;1;0;0] #thus e'v=1

    M = e.* (uss - ∇u'*vss .+ (0.5 .* vss' *Hu *vss) ) * e' +
        0.5 *(∇u * e' - e * vss' * Hu - Hu*vss*e' +e* ∇u') +
        0.5 * Hu


    #Translating M into the Matrices we need:
    Q = M[1:3,1:3]
    W = M[1:3,4:5]
    R = M[4:5,4:5]


    A = [0 0 0 ;0 ρ 0 ; 0 0 1]
    B = [1  0 ;  0 0;0 0]
    C = [0 ; 1 ; 0]

    #X_t+1 = AX_t + Bu_t + Cϵ_t+1

    #Mapping to the problem without discounting (1 VARIABLES ARE ~ IN LECTURE NOTES)
    A1 = sqrt(β) *(A -B* (R\ W'))
    B1 = sqrt(β) *B
    Q1 = Q- W*(R\W')


    P, F = Riccati(A1,B1,Q1,R,W,β)

    F1= F-inv(R) *W';F[:,3] = F[:,3]/2;P[3,:]=1.41*P[3,:];



return A,B,C,P ,F,A1,B1, F1,kss,hss
end
