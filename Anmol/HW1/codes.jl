#This is where I actually code the homework stuff, refer to the Jupyter notebook for a better presented version :)

using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots
using DataFrames

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
#vector is a vector with [s,n,k,Cn,Cs,Ck,Θn,Θs,Θk]

function shimerT1(vector::Vector)
    s,n,k,Cn,Cs,Ck,Θn,Θs,Θk = vector
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
    s,n,k,Cn,Cs,Ck,Θn,Θs,Θk = vector
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

    T[1:3]=ForwardDiff.gradient(shimerT1,[sbar,nss,kss,Cn,Cs,Ck,Θn,Θs,Θk])[1:3]
    T[4:6]=ForwardDiff.gradient(shimerT2,[sbar,nss,kss,Cn,Cs,Ck,Θn,Θs,Θk])[1:3]
end

coefs = nlsolve(loglin!, [0.929703, 0.014, 0.776461, 0.754842, 7.38, 0.962334],ftol = :1.0e-9, method = :trust_region , autoscale = true)

Cn,Cs,Ck,Θn,Θs,Θk = coefs.zero
C(s,n,k) = C(s,n,k,sbar,nss,kss,Css,Cn,Cs,Ck)
Θ(s,n,k) = Θ(s,n,k,sbar,nss,kss,Θss,Θn,Θs,Θk)

    function findK(vector::Vector)
        k1,s,n,k = vector
        c=C(s,n,k)
        θ=Θ(s,n,k)
        s1 = (1-ρ)*sbar +ρ*s
        eq = k1 - exp(-s1/(1-α)) * ((k)^α * (n-θ*(1-n))^(1-α) + (1-δ)*k - c)
        return eq
    end

T=ForwardDiff.gradient(findK,[kss,sbar,nss,kss])
kn, kk,ks= [-nss*T[3]/(kss*T[1]), -kss*T[4]/(kss*T[1]),-exp(sbar)*T[2]/(kss*T[1])]

k1(s,n,k) = exp(log(kss)+ks*(s-sbar)+ kn*(log(n)-log(nss))+kk*(log(k)-log(kss)))

function findn(vector::Vector)
    n1,s,n,k = vector
    θ=Θ(s,n,k)
    eq = n1 -( (1-x)*n+f(θ)*(1-n))
    return eq
end

T=ForwardDiff.gradient(findn,[nss,sbar,nss,kss])
nn,nk,ns= [-nss*T[3]/(nss*T[1]),-kss*T[4]/(nss*T[1]),-exp(sbar)*T[2]/(nss*T[1])]
n1(s,n,k) = exp(log(nss)+ks*(s-sbar)+ nn*(log(n)-log(nss))+nk*(log(k)-log(kss)))
#the other variables for the correlation Matrix
function Y(vector::Vector)
    y,s,n,k = vector
    θ=Θ(s,n,k)
    eq = -y + k^α *exp(s)*(n-θ*(1-n))^(1-α)
    return eq
end

yss = Y([0,sbar,nss,kss])
T=ForwardDiff.gradient(Y,[yss,sbar,nss,kss])
yn,yk,ys= [-nss*T[3]/(yss*T[1]),-kss*T[4]/(yss*T[1]),-(sbar)*T[2]/(yss*T[1])]
y(s,n,k)=exp(log(yss)+ys*(s-sbar)+ yn*(log(n)-log(nss))+yk*(log(k)-log(kss)))

function CY(vector::Vector)
    cy, s,n,k = vector
    eq = -cy + C(s,n,k)/y(s,n,k)
    return eq
end
cyss = CY([0,sbar,nss,kss])
T=ForwardDiff.gradient(CY,[cyss,sbar,nss,kss])
cyn,cyk,cys= [-nss*T[3]/(cyss*T[1]),-kss*T[4]/(cyss*T[1]),-exp(sbar)*T[2]/(cyss*T[1])]
CY(s,n,k)=exp(log(cyss)+cys*(s-sbar)+ cyn*(log(n)-log(nss))+cyk*(log(k)-log(kss)))

function Nwedge(vector::Vector)
    τ, s,n,k=vector
    cy = CY(s,n,k)
     eq = 1- (γ/(1-α)) * (cy) * n -τ
    return eq
end

τss = Nwedge([0,sbar,nss,kss])
T=ForwardDiff.gradient(Nwedge,[τss,sbar,nss,kss])
τn,τk,τs= [-nss*T[3]/(τss*T[1]),-kss*T[4]/(τss*T[1]),-exp(sbar)*T[2]/(τss*T[1])]
Nwedge(s,n,k)=exp(log(τss)+τs*(s-sbar)+ τn*(log(n)-log(nss))+τk*(log(k)-log(kss)))

function WNy(vector::Vector)
    wny, s,n,k=vector
    c=C(s,n,k)
    θ=Θ(s,n,k)
    w = (1/(1-τ)) * ((1-φ)*γ * c + φ*(1-τ)*(1-α)*(k/(n-θ*(1-n)))^α *(1+θ))

    eq = -wny + w*(n/y(s,n,k))

    return eq
end

wnyss = WNy([0,sbar,nss,kss])
T=ForwardDiff.gradient(WNy,[wnyss,sbar,nss,kss])
wnyn,wnyk,wnys= [-nss*T[3]/(wnyss*T[1]),-kss*T[4]/(wnyss*T[1]),-exp(sbar)*T[2]/(wnyss*T[1])]
wny(s,n,k)=exp(log(wnyss)+wnys*(s-sbar)+ wnyn*(log(n)-log(nss))+wnyk*(log(k)-log(kss)))
A=[ρ 0 0;
ns nn nk;
ks kn kk ]


D= [ζ;0; 0]

Σ = ones(3,3)
d=10
while d>10^(-15)
    global Σ, d
    Σ1 = A*Σ*A' + D*D'
    d = maximum(abs.(Σ-Σ1))
    Σ=Σ1
end

Atilde=[ ys yn yk;
Cs Cn Ck;
Θs Θn Θk ;
ks kn kk ;
ns nn nk ;
wnys wnyn wnyk;
cys cyn cyk;
τs τn τk;
ρ 0 0]

Covar = Atilde*Σ*Atilde'
relative_std=zeros(size(Covar)[1])
corr = UpperTriangular(zeros(size(Covar)))
for i=1:size(Covar)[1]
    global corr, relative_std
    for j = i:size(Covar)[1]
    corr[i,j] = Covar[i,j]/(sqrt(Covar[i,i])*sqrt(Covar[j,j]))
    end
    relative_std[i] = sqrt(Covar[i,i])/sqrt(Covar[1,1])
end

table=vcat(relative_std',corr)

m(s,n,k)=[s-sbar,log(n)-log(nss),log(k)-log(kss)]


E=zeros(3,120)
M=ones(3,120).*m(sbar+ζ/(1-ρ^2)^(1/2),nss,kss)
devs=zeros(120).+ζ/(1-ρ^2)^(1/2)
devk=zeros(120)
devn=zeros(120)
devC=zeros(120)
devY=zeros(120)
devτ=zeros(120)
devwny=zeros(120)
devcy=zeros(120)
devΘ=zeros(120)
for t=1:120
if t>1
    devs[t] = ρ*devs[t-1]
    devk[t] = ks*devs[t-1]+kn*devn[t-1]+kk*devk[t-1]
    devn[t] = ns*devs[t-1]+nn*devn[t-1]+nk*devk[t-1]
end
    devC[t] = Cs*devs[t]+Cn*devn[t]+Ck*devk[t]
    devY[t] = ys*devs[t]+yn*devn[t]+yk*devk[t]
    devτ[t] = τs*devs[t]+τn*devn[t]+τk*devk[t]
    devwny[t] = wnys*devs[t]+wnyn*devn[t]+wnyk*devk[t]
    devcy[t] = cys*devs[t]+cyn*devn[t]+cyk*devk[t]
    devΘ[t] =   Θs*devs[t]+Θn*devn[t]+Θk*devk[t]
end
<<<<<<< HEAD


plot(
    plot(devs,legend = false,ylabel="Productivity Growth s"),
    plot(devY.+devs,legend = false,ylabel="Output y"),
    plot(devC.+devs,legend = false,ylabel="Consumption c"),
    plot(devk.+devs,legend = false,ylabel="Capital k"),
=======

plot(
    plot(devs,legend = false,ylabel="Productivity Growth s"),
    plot(devY,legend = false,ylabel="Output y"),
    plot((devC),legend = false,ylabel="Consumption c"),
    plot(devk,legend = false,ylabel="Capital k"),
>>>>>>> Anmol-HW1-IRFs
    plot(devn,legend = false,ylabel="Employment n"),
    plot(devΘ,legend = false,ylabel="Recruiting/Employment"),
    plot(devτ,legend = false,ylabel="Labor Wedge"),
    plot(devwny,legend = false,ylabel="Labor Share"),
    plot(devcy,legend = false,ylabel="C/Y"),
)
<<<<<<< HEAD
=======


#Working with Data

using CSV
using Dierckx #Interpolation package, I did not understand the package Interpolations
using Dates
#Loading Data
Kdata = CSV.read("RKNANPUSA666NRUG.csv")
Ndata = CSV.read("labor.csv")
Ydata = CSV.read("GDPC1.csv")

#Renaming Columns
rename!(Kdata,  [:RKNANPUSA666NRUG => :CapitalStock])
rename!(Ndata,  [Symbol("hours/population") => :Labor])
rename!(Ydata,  [:GDPC1 => :GDP])

#Same starting and ending year
Ndata = Ndata[Kdata[:DATE][1].<=Ndata[:DATE].<=Kdata[:DATE][end],:]
Ydata = Ydata[Kdata[:DATE][1].<=Ydata[:DATE].<=Kdata[:DATE][end],:]

#DATA will be a DataFrame containing all variables we buiild it from Ndata since it is monthly
DATA = copy(Ndata)
#We will use logs only
DATA[:Labor] = log.(DATA[:Labor])
DATA[:GDP] = log.(Ydata[:GDP]./DATA[:Population])

#Now, we need to interpolate Kdata from anual to quarterly

interpK = Spline1D(Dates.value.(Kdata[:DATE]), Kdata[:CapitalStock])
#Interpolate and get the logs (percapita):
DATA[:CapitalStock] = log.(interpK(Dates.value.(Ndata[:DATE]))./DATA[:Population])

#Now, we have our dataset. We need to use the HP filter to remove trends
using QuantEcon

DATA[:Labor_dev],DATA[:Labor_trend] = hp_filter(DATA[:Labor],1600)
DATA[:GDP_dev],DATA[:GDP_trend] = hp_filter(DATA[:GDP],1600)
DATA[:CapitalStock_dev],DATA[:CapitalStock_trend] = hp_filter(DATA[:CapitalStock],1600)

#To get the log deviations of the TFP shock and trend:

DATA[:TFPdev] = (DATA[:GDP_dev] - yn*DATA[:Labor_dev]-yk*DATA[:CapitalStock_dev])./ys
DATA[:TFPtrend] = (DATA[:GDP_trend] - yn*DATA[:Labor_trend]-yk*DATA[:CapitalStock_trend])./ys
DATA[:TFP] = DATA[:TFPtrend]+DATA[:TFPdev]

#Plots!
plot(
    plot([DATA[:TFP],DATA[:TFPtrend]],legend = :bottomright, label = ["TFP","TREND"]),
    plot([DATA[:TFPdev]],legend = :bottomright, label = ["TFP log deviations"])

)
>>>>>>> Anmol-HW1-IRFs
