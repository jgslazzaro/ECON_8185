#Author: Joao Lazzaro
#This script calls the VFI functions and plots the policy and Value functions
include("VFI.jl")

using Printf
using Plots

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

#grid for k

KVFI = 150 #number of gridpoints for capital
kmax = K[end] # maximum value for k (set low if no productivity shocks!)
kmin = K[1]+eps() #minimum value for k
#grid for z
Z = 1 #number of productivity states

V, policy_k,policy_h,policy_c,k,h, z, Π = VFI(δ,θ,β,ρ,σ,μ,γn,γz,ϕ,KVFI,kmax,kmin,Z,Xnneg)
