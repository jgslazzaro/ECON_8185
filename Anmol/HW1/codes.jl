#This is where I actually code the homework stuff, refer to the Jupyter notebook for a better presented version :)

using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots

#Parameters
β = 0.996              #Discount Rate
sbar = 0.0012          #Mean productivity growth
τ = 0.4                #Tax Rate
x = 0.034              #Employment exit Probability
f(θ) = 2.32 .*θ.^(1/2)  #Matching Function
φ = 0.5                #Worker's bargaining power
γ = 0.471              #Disutility of Work
α = 0.33               #Capital Share
δ = 0.0028             #Depreciation
ρ = 0.4               #Autocorrelation of productivity
ζ = 0.00325              #Standard Deviation of productivity
μ(θ) = f(θ)/θ          #Hiring rate per Vacancy


#Create a function to find the SS


# U is a vector with (Θ,C,k,n)
function SS!(eq, U)
    Θ = U[1]
    C = U[2]
    k = U[3]
    n = U[4]
    #Capital law of Motion
    eq[1] = k*exp(sbar/(1-α)) - ((k)^α * (n-Θ*(1-n))^(1-α) + (1-δ)*k - C)
    #Employment law of motion
    eq[2] = n - ((1-x)*n+f(Θ)*(1-n))
    #Equation 5
    eq[3] = (1-α)*(k/(n-Θ*(1-n)))^α +
    - β *μ(Θ)*((-(1-φ)*γ*C/(1-τ))+ (1-α)*(k/(n-Θ*(1-n)))^α *((1-x)/μ(Θ) +1-φ-φ*Θ) )
    #Equation 1
    eq[4] = 1 - exp(-sbar/(1-α))*(β * (α*(k/(n-Θ*(1-n)))^(α-1)+1-δ))
end

#Newton, nonlinear Solver
SS = nlsolve(SS!, [0.07,4.7,218,0.95],ftol = :1.0e-9, method = :trust_region , autoscale = true)
Θss,Css,kss,nss = SS.zero


#Consumption function:

function C(s,n,k,sbar,nss,kss,Css,Cn,Cs,Ck)
    C=exp(log(Css)+Cs*(s-sbar)+ Cn*(log(n)-log(nss))+Ck*(log(k)-log(kss)))
    return C
end
# Θ Function
function Θ(s,n,k,sbar,nss,kss,Θss,Θn,Θs,Θk)
    Θ=exp(log(Θss)+Θs*(s-sbar)+ Θn*(log(n)-log(nss))+Θk*(log(k)-log(kss)))
    return Θ
end
function Sdeterministic(s,ρ,sbar)
    s1 = (1-ρ)*sbar +ρ*s
    return s1
end

#SHimer's method to find a Log-Linear approximation. Let T(k,n,s)=0 be the equilibrium equations
#Then, plugging:
# ln(θ)-ln(θss) = θs(s-bars)+θk(ln(k)-ln(kss))+θn(ln(n)-ln(nss))
# ln(c)-ln(css) = cs(s-bars)+ck(ln(k)-ln(kss))+cn(ln(n)-ln(nss))
#Into T and setting its derivatives to zero, we should get a system of equations that
#determines the coefficients.
#vector is a vector with [s,n,k,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk]

function shimerT1(vector::Vector)
    s,n,k,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk = vector
    c=C(s,n,k,sbar,nss,kss,Css,Cn,Cs,Ck)
    θ=Θ(s,n,k,sbar,nss,kss,Θss,Θn,Θs,Θk)
    s1 = Sdeterministic(s,ρ,sbar)
    k1 = exp(-s1/(1-α)) * ((k)^α * (n-θ*(1-n))^(1-α) + (1-δ)*k - c)
    n1 = (1-x)*n+f(θ)*(1-n)
    c1=C(s1,n1,k1,sbar,nss,kss,Css,Cn,Cs,Ck)
    θ1 = Θ(s1,n1,k1,sbar,nss,kss,Θss,Θn,Θs,Θk)

    eq = β* c/c1 * exp(-s1/(1-α)) *(α  *(k1/(n1-θ1*(1-n1)))^(α-1)+1-δ) - 1

    return eq
end


function shimerT2(vector::Vector)
    s,n,k,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk = vector
    c=C(s,n,k,sbar,nss,kss,Css,Cn,Cs,Ck)
    θ=Θ(s,n,k,sbar,nss,kss,Θss,Θn,Θs,Θk)
    s1 = Sdeterministic(s,ρ,sbar)
    k1 = exp(-s1/(1-α)) * ((k)^α * (n-θ*(1-n))^(1-α) + (1-δ)*k - c)
    n1 = (1-x)*n+f(θ)*(1-n)
    c1=C(s1,n1,k1,sbar,nss,kss,Css,Cn,Cs,Ck)
    θ1 = Θ(s1,n1,k1,sbar,nss,kss,Θss,Θn,Θs,Θk)


    eq = -(1-α)*(k/(n-θ*(1-n)))^α + β* μ(θ) * (c/c1) *(-(1-φ)*γ*c1/(1-τ) + (1-α) *(k1/(n1-θ1*(1-n1)))^α *((1-x)/μ(θ1)+1-φ-φ*θ1))

    return eq
end

#coefs is a vector of unknown coefficients
function loglin!(T,coefs::Vector)
    Cn,Cs,Ck,Θn,Θs,Θk = coefs

    T[1:3]=ForwardDiff.gradient(shimerT1,[sbar,nss,kss,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk])[1:3]
    T[4:6]=ForwardDiff.gradient(shimerT2,[sbar,nss,kss,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk])[1:3]
end

coefs = nlsolve(loglin!, [0.929703, 0.014, 0.776461, 0.754842, 7.38, 0.962334],ftol = :1.0e-9, method = :trust_region , autoscale = true)

Cn,Cs,Ck,Θn,Θs,Θk = coefs.zero


    function findK(vector::Vector)
        k1,s,n,k,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk = vector
        c=C(s,n,k,sbar,nss,kss,Css,Cn,Cs,Ck)
        θ=Θ(s,n,k,sbar,nss,kss,Θss,Θn,Θs,Θk)
        s1 = (1-ρ)*sbar +ρ*s
        eq = k1 - exp(-s1/(1-α)) * ((k)^α * (n-θ*(1-n))^(1-α) + (1-δ)*k - c)
        return eq
    end

T=ForwardDiff.gradient(findK,[kss,sbar,nss,kss,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk])[1:4]

kn, kk,ks= [-nss*T[3]/(kss*T[1]), -kss*T[4]/(kss*T[1]),-exp(sbar)*T[2]/(kss*T[1])]


function findn(vector::Vector)
    n1,s,n,k,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk = vector
    θ=Θ(s,n,k,sbar,nss,kss,Θss,Θn,Θs,Θk)

    eq = n1 -( (1-x)*n+f(θ)*(1-n))
    return eq
end

T=ForwardDiff.gradient(findn,[nss,sbar,nss,kss,sbar,nss,kss,Css,Θss,Cn,Cs,Ck,Θn,Θs,Θk])[1:4]
nn,nk,ns= [-nss*T[3]/(nss*T[1]),-kss*T[4]/(nss*T[1]),-exp(sbar)*T[2]/(nss*T[1])]


D= [ζ; 0; 0]
A=[ρ 0 0;
ns nn nk;
ks kn kk]

eigvals(A)



#Defining K and N
function K1(s,n,k,sbar,nss,kss,ks,kn,kk)
    level=exp(log(kss)+ks*(s-sbar)+kn*(log(n)-log(nss)) + kk*(log(k)-log(kss)))
    return level
end
function N1(s,n,k,sbar,nss,kss,ns,nn,nk)
    level=exp(log(nss)+ks*(s-sbar)+nn*(log(n)-log(nss)) + nk*(log(k)-log(kss)))
    return level
end

A=[ρ 0 0;
ns nn nk;
ks kn kk]

U = [ Θs Θn Θk; Cs Cn Ck]


if maximum(eigvals(A)) >= 1
    error("Eigenvalue > 1")
end
D= [ζ; 0; 0]
m(s,n,k)=[s-sbar,log(n)-log(nss),log(k)-log(kss)]
mtilde(s,n,k)=[s-sbar,log(n)-log(nss),log(k)-log(kss),log(Θ(s,n,k,sbar,nss,kss,Θss,Θn,Θs,Θk))-log(Θss),
log(C(s,n,k,sbar,nss,kss,Css,Cn,Cs,Ck))-log(Css)]
Atilde = hcat(vcat(A,U),zeros(5,2))

Σ = ones(3,3)
d=10
while d>10^(-15)
    global Σ, d
    Σ1 = A*Σ*A' + D*D'
    d = maximum(abs.(Σ-Σ1))
    Σ=Σ1
end

#covar = Atilde*Σ*Atilde'

chur=schur(A)

T = chur.T
Z=chur.Z



E=zeros(3,120)
M=ones(3,120).*m(sbar+ζ/(1-ρ^2)^(1/2),nss,kss)
E[:,1]=inv(Z)*M[:,1]


plot(M[1,:].*100)
