#Author: Joao Lazzaro
#This script calls the VFI functions and plots the policy and Value functions
include("VFI.jl")

using Printf
using Plots

δ = 1    #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
ρ = 0.5  #AR coefficient
σ = 0.5  #AR shock SD
μ = 0    #AR(1) constant term
ϕ = 0    #Labor parameter
γn= 0    #Population growth rate
γz= 0    #Productivitu growth rate

Xnneg = true #Non negative investment

#grid for k

K = 15 #number of gridpoints for capital
kmax = 1/5 # maximum value for k (set low if no productivity shocks!)
kmin = 0.01 #minimum value for k
#grid for z
Z = 2 #number of productivity states

Tx = 2 #Number of Tx states
Th = 2 #Number of Th States
G  = 2 #Number of G States

V, policy_k,policy_h, k,h, z, Π = VFI(δ,θ,β,ρ,σ,μ,γn,γz,ϕ,K,kmax,kmin,Z,Xnneg);

#plot the Value function griven 2 level of z
labels = [@sprintf("z = %.2f",z[2]),@sprintf("z = %.2f",z[9]) ]
data = [V[:,2],V[:,9]]
plot(k, data, label = labels)


#plot the capital policy function given 2 level of z
labels = [@sprintf("z = %.2f",z[2]),@sprintf("z = %.2f",z[9]) ]
data = [k[policy_k[:,2]],k[policy_k[:,9]]]
plot(k, data, label = labels)

#plot the Labor policy function given 2 level of z. Note, labor choice is constant!
labels = [@sprintf("z = %.2f",z[2]),@sprintf("z = %.2f",z[9]) ]
data = [h[policy_h[:,3]],h[policy_h[:,9]]]
plot(k, data, label = labels)
