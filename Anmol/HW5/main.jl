include("functions.jl")
include("Reiter.jl")
include("AgentProblem.jl")
cd("Z:\\ECON_8185\\Anmol\\HW5")
#Defining parameters they were taken frmom KS
const α = 0.36 #capital share in output
const β = 0.99 #Impatience
const δ = 0.025 #Depreciation rate
const η = 1.0/2.9 #labor /consumption elasticity
const μ = 1.0 #Intertemporal elasticity
const A= 1.0#(1/β-1+δ)/α
const lbar = 1.0#/0.9 #time endowment

const Ap_size = 30 #number of assets gridpoints for policy rules
const Ad_size = 150 #number of assets gridpoints for distribution
amin = 0.0
amax = 50.0

factor =7
const Ap =(range(0.0, stop=Ap_size-1, length=Ap_size)/(Ap_size-1)).^factor * amax #Capital grid for todayconst A1 = A#(range(0, stop=nA-1, length=nA)/(nA-1)).^factor * amax #Capital grid for today#range(0., stop=amax, length=nA).^1 #Capital grid for tomorrow
const Ad = range(0.0, stop=amax, length=Ad_size).^1.0 #Capital grid for todayconst A1 = A#(range(0, stop=nA-1, length=nA)/(nA-1)).^factor * amax #Capital grid for today#range(0., stop=amax, length=nA).^1 #Capital grid for tomorrow

#Employment shocks
#const E = [0.0,1.0]
#const E_size = length(E)

update = 0.2

uDuration = 1                                                               # number of period unemployed before employed
aggEmployment = .93 #Aggregate labor supply in an exogenous labor choice setting, for comparison
    #== Transition Probabilities ==#
#pdfE = zeros(E_size,E_size)
#pdfE[1,:] = [uDuration/(1 + uDuration) 1-(uDuration / (1 + uDuration))]
#pdfE[2,1] = ((1 - aggEmployment) / aggEmployment) * (1 - (uDuration / (1 + uDuration)))
#pdfE[2,2] = 1-((1 - aggEmployment) / aggEmployment) * (1 - (uDuration / (1 + uDuration)))


#pdfE = [0.0 1.0; 0.0 1.0]#[0.05 0.95; 0.02 0.98]

#productivity shocks
ρ = 0.8
σ = 0.01
Z1(z;ρ=ρ,σ =σ) = exp(ρ * log(z) + σ * randn())
zss=1.0

E_size = 5 #Number of states for e
σe = 0.3 #Variance
ρe = 0.6
pdfE,E = Tauchen(ρe,σe,E_size,1-ρe)    #E comes from Tauchen method

#Initial guess for wages
w0 = 2.3739766754889238
R0=1.010012230693079

#First we fin the State States

policygridss,Kss,Lss,Rss,wss,λss = findSS(Ap,Ad,E,E_size,Ad_size,pdfE;tol = 1e-6,update = .1,R0 = R0,w0=w0)

sum(λ)
using Plots
plot(Ad,sum(λ,dims=2))

using JLD2, FileIO
@save "SteadyState.jld2" policygridss Kss Lss Rss wss λss
@load "SteadyState.jld2"
Kss,Lss,Rss,wss,λss = K0,L0,R0,w0,λ


    #LAW OF MOTIONS
function λ1(λ,K,L,Z;R=R,w=w,policy_a=policy_a)
    weight = weightmatrix(policy_a,R,w)
    lambda = weight * λ
return lambda
end

function K1(λ,K,L,Z;R=R,w=w,policy_a=policy_a,Ad=Ad)
    Knew = sum(policy_a.(Ad,E[1]).*λ[:,1] .+ policy_a.(Ad,E[2]).*λ[:,2])
    return Knew
end

function L1(λ,K,L,Z;policy_n=policy_n,Ad=Ad)
    L0 = sum(policy_n.(Ad,E[1]).*λ[:,1] .+ policy_n.(Ad,E[2]).*λ[:,2])
    return L0
end

Z1(z;ρ=ρ,σ =σ) = exp(ρ * log(z) + σ * randn())
function pol(λ,K,L,Z;pss=policygridss,λss=λss,Kss=Kss,Lss=Lss,Zss=zss,pλ=pλ,pK=pK,pL=pL,pZ=pZ)
    #pλ is a matrix ApxAd
    poli =exp.(log.(pss) .+ pK*(log(K).-log(Kss)) .+ pL*(log(L).-log(Lss)) .+ pZ*(log(Z).-log(Zss)) .+ pλ*(log(λ).-log(λss)))

    return poli
end

function lR(λ,K,L,Z;Rss=Rss,λss=λss,Kss=Kss,Lss=Lss,Zss=zss,pλ=pλ,pK=pK,pL=pL,pZ=pZ)
    #Rλ is a matrix 1xAd
    poli =exp.(log.(Rss) .+ RK*(log(K).-log(Kss)) .+ RL*(log(L).-log(Lss)) .+ RZ*(log(Z).-log(Zss)))

    return poli
end

function lw(λ,K,L,Z;wss=wss,λss=λss,Kss=Kss,Lss=Lss,Zss=zss,pλ=pλ,pK=pK,pL=pL,pZ=pZ)
    #Rλ is a matrix 1xAd
    poli =exp.(log.(wss) .+ wK*(log(K).-log(Kss)) .+ wL*(log(L).-log(Lss)) .+ wZ*(log(Z).-log(Zss)))

    return poli
end


function system(λ,K,L,Z;pdfE=pdfE,E=E,policygridss=policygridss,Kss=Kss,Lss=Lss,
    Rss=Rss,wss=wss,λss=λss,ρ=ρ,σ =σ,α=α,β=β,δ=δ,Ap=Ap)
    R = R(λ,K,L,Z
    w=  w(λ,K,L,Z)
    policygrid = pol(λ,K,L,Z)
    itpc = LinearInterpolation((Ap,E),pol(λ,K,L,Z), extrapolation_bc=Line())
    policy_c(a,e) = itpc(a,e)
    policy_n(a,e) = nstar(policy_c(a,e),e,w)
    policy_a(a,e;w=w,R=R) = a1star(policy_c(a,e),a,e,w,R)

    lambdanew = λ1(λ,K,L,Z;R=R,w=w,policy_a=policy_a)
    Knew = K1(λ,K,L,Z;R=R,w=w,policy_a=policy_a,Ad=Ad)
    Lnew = L1(λ,K,L,Z;policy_n=policy_n,Ad=Ad)
    Znew = Z1(z;ρ=ρ,σ =σ)

    return Rp,wp,policygrid,lambda1,Knew,Lnew,Znew
end

function polgrid(states;Ap=Ap,Ad=Ad,E=E,pdfE=pdfE,E_size=E_size)
    K,L,Z = states[end-2:end]
    λ = reshape(states[1:end-3],length(Ad),E_size)
    Rp = R(Z,K,L)
    wp = w(Z,K,L)
    pgrid = ENDOGENOUSGRID(Rp,wp,Ap,Ap,E,pdfE)
return pgrid[:]
end

using ForwardDiff
ForwardDiff.jacobian(polgrid, vcat(λss[:],Kss,Lss,zss))

states = vcat(λss[:],Kss,Lss,zss)
bla = polgrid(vcat(λss[:],Kss,Lss,zss))

#WAGES
Rp = R(Z,K,L)
wp = w(Z,K,L)

#getting policyfunctions
policygrid = ENDOGENOUSGRID(Rp,wp,Ap,A1p,E,pdfE)
itpc = LinearInterpolation((Ap,E),policygrid, extrapolation_bc=Line())
policy_c(a,e) = itpc(a,e)
policy_n(a,e) = nstar(policy_c(a,e),e,w)
policy_a(a,e;w=wp,R=Rp) = a1star(policy_c(a,e),a,e,w,R)

lambda1 = λ1(λ,K,L,Z;R=Rp,w=wp,policy_a=policy_a)
Knew = K1(λ,K,L,Z;R=Rp,w=wp,policy_a=policy_a)
Lnew = L1(λ,K,L,Z;policy_n=policy_n,Ad=Ad)
Znew = Z1(z;ρ=ρ,σ =σ)
