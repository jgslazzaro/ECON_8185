﻿
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
#cd("\\\\tsclient\\C\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
include("functions.jl")
using  LinearAlgebra, Plots
using  JLD2,FileIO


#Defining Parameters:
#I'll use a similar calibration as KS
β = 0.99 #Discount rate
μ = 1  #Elasticity of intertemporal substitution
η = 1 #Utility parameter
δ = 0.075 #Depreciation rate
θ = 0.3 #Capital Share of output


ρe = 0.6 #autocorrelation of employment shock
σe = 0.3 #Variance of employment shock
ρz = 0.6 #autocorrelation of Aggregate shock
σz = 0.2 #Variance of Aggregate shock


fast = false #Used to get the policy functions faster, assuming we know the
#equilibrium r and w

amin = -0.0   #Debt limit
amax= 13.0 #capital limit

nE = 2 #Number of states for e
nZ = 2 #Number of states for e
nA = 30#states for assets

pdfZ,Z = Tauchen(ρz,σz,nZ)    #Z comes from Tauchen method
Z = exp.(Z) #Z is nonnegative

E = [0.0 1.0]

pdfE = ones(nE,nE,nZ)

pdFE[:,:,1] = [0.1 0.9; 0.1 0.9]
pdfE[:,:,2] = [0.04 0.96; 0.04 0.96]


##

#Defining grids

#pdfE,E = Tauchen(ρe,σe,nE)    #E comes from Tauchen method
#pdfZ,Z = Tauchen(ρz,σz,nZ)    #Z comes from Tauchen method
#Z = exp.(Z) #Z is nonnegative
A = range(amin,stop = amax, length = nA) #Half points will be in the first third
# of the grid


#Initial guesses:

r= ((1/β - 1)-0.0001).*ones(nZ) #initial guess for r liens in the interval (-δ, 1/β-1)

Kguess = ((r.+δ)./(Z.*θ)).^(1/(θ-1))#K for the nitial guess of r
w=  (1-θ).*Z.*Kguess.^θ  #Initial wage given r0
Nguess = (w./(Z.*(1-θ))).^(-1/θ).*Kguess

λ,r,w, policy_a, policy_c, policy_l,Assets,N,Y,B,K = ayiagary(A,E,r,w,τy,T,β,η,μ,Z,G,fast)