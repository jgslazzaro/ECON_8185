include("functions.jl")
include("Reiter.jl")
#Defining parameters they were taken frmom KS
const α = 0.36 #capital share in output
const β = 0.99 #Impatience
const δ = 0.025 #Depreciation rate

const η = 1.0/2.9 #labor /consumption elasticity
const μ = 1.0 #Intratemporal elasticity

const lbar = 1.0#/0.9 #time endowment

const Ap_size = 100 #number of assets gridpoints for policy rules
const Ad_size = 200 #number of assets gridpoints for distribution
amin = 0.0
amax = 30.0

factor =3.5
const Ap =(range(0, stop=Ap_size-1, length=Ap_size)/(Ap_size-1)).^factor * amax #Capital grid for todayconst A1 = A#(range(0, stop=nA-1, length=nA)/(nA-1)).^factor * amax #Capital grid for today#range(0., stop=amax, length=nA).^1 #Capital grid for tomorrow
const Ad = range(0, stop=amax, length=Ad_size) #Capital grid for todayconst A1 = A#(range(0, stop=nA-1, length=nA)/(nA-1)).^factor * amax #Capital grid for today#range(0., stop=amax, length=nA).^1 #Capital grid for tomorrow

#Employment shocks
const E = [0.0,1.0]
const E_size = length(E)

pdfE = [0.1 0.9; 0.05 0.95]


#productivity shocks
ρ = 0.8
σ = 0.01
Z1(z;ρ=ρ,σ =σ) = exp(ρ * log(z) + σ * randn())

const nd_size = 20
const nd = range(0,stop = lbar,length  = nd_size)


#SS level of z
zss = exp(0.0)


#initial guess for K and L
K0 = 25.0
L0= 0.5


#wages given guesses:
R0 = R(zss,K0,L0)
w0 = w(zss,K0,L0)



#Find policy function given wages:
policygrid = ENDOGENOUSGRID(R0,w0,Ap,Ap,E,pdfE)
itpc = LinearInterpolation((Ap,E),policygrid, extrapolation_bc=Line())
policy_c(a,e) = itpc(a,e)
policy_n(a,e;w=w0) = nstar(policy_c(a,e),e,w)
policy_a(a,e;w=w0,R=R0) = a1star(policy_c(a,e),a,e,w,R)

using Plots
plot(Ap,policy_a.(Ap,.0),legend=:bottomright,label ="e=0")
plot!(Ap,policy_a.(Ap,1.0),label ="e=1")
plot!(Ap,Ap,label ="45")

plot(Ap,policy_c.(Ap,.0),legend=:bottomright,label ="e=0")
plot!(Ap,policy_c.(Ap,1.0),label ="e=1")

#Find the invariant distribution implied by policy functions:
λ = fill(1/(Ad_size*E_size),Ad_size,E_size)
λ = invariantλ(λ,policy_a)



sum(λ)

#Aggregate capital and labor
K2 = sum(Ad.*λ)
L2 = sum([policy_n(Ad[ai],e)*λ[ai,ei] for ai=1:length(Ad),(ei,e) in enumerate(E)])

distance = max(abs(K0-K2),abs(L0-L2))




sum([poln(Ad[ai],e)*λ[ai,ei] for ai=1:length(Ad),(ei,e) in enumerate(E)])







using Plots

plot(Ad,sum(λ,dims=2),yaxis=[0.0,0.05])
