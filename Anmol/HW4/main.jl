

include("functions.jl")
using  LinearAlgebra, Plots
using  JLD2,FileIO


#Defining Parameters:
#I'll use a similar calibration as KS
β = 0.99 #Discount rate
μ = 1  #Elasticity of intertemporal substitution
η = 1.9/2.9 #Utility parameter
δ = 0.025 #Depreciation rate
θ = 0.36 #Capital Share of output


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
nA = 30 #states for assets
nK = 10 #states for aggregate Capital
nL = 10 #states for aggregate Labor
pdfZ,Z = Tauchen(ρz,σz,nZ)    #Z comes from Tauchen method
Z = exp.(Z) #Z is nonnegative

E = [0.0 1.0]

pdfE = ones(nE,nE,nZ)

pdfE[:,:,1] = [0.1 0.9; 0.1 0.9]
pdfE[:,:,2] = [0.04 0.96; 0.04 0.96]

##

#Defining grids

#pdfE,E = Tauchen(ρe,σe,nE)    #E comes from Tauchen method
#pdfZ,Z = Tauchen(ρz,σz,nZ)    #Z comes from Tauchen method
#Z = exp.(Z) #Z is nonnegative
A = range(amin,stop = amax, length = nA) #Half points will be in the first third
K = copy(A)
L = range(0,stop = 1, length = NL)
# of the grid


#Initial guesses:



λ,r,w, policy_a, policy_c, policy_l,Assets,N,Y,B,K = KrusselSmith(A,E,Z,K,L,pdfE,pdfZ)
