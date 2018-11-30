
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
#cd("\\\\tsclient\\C\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
include("functions.jl")
using  LinearAlgebra, Plots
using  JLD2,FileIO


#Defining Parameters:
β = 0.98 #Discount rate
μ = 1.5  #Elasticity of intertemporal substitution
η = 0.3 #Utility parameter
τy = 0.4 #Income tax
ρ = 0.6 #autocorrelation
σ = 0.3 #Variance
δ = 0.075 #Depreciation rate
θ = 0.3 #Capital Share
T=0.1 #tax rate, to be added
Z=16.33#productivity level (not used)

b = -0.0   #Debt limit
amax=.3  #capital limit

nE = 5 #Number of states for e
nA = 500 #states for assets


r= (1/β - 1)*rand() #initial guess for r liens in the interval (-δ, 1/β-1)

Kguess = ((r+δ)/(Z*θ))^(1/(θ-1))#K for the nitial guess of r
w=  (1-θ)*Z*Kguess^θ  #Initial wage given r0
Nguess = (w/(Z*(1-θ)))^(-1/θ)*Kguess
##

#Defining grids

pdfE,E = Tauchen(ρ,σ,nE)    #E comes from Tauchen method
A = range(b,stop = amax/3, length = Int64(nA/2)) #Half points will be in the first third of the grid
A  =vcat(A, range(A[end],stop = amax, length = nA-Int64(nA/2)))  #Assets


dist_y = 1
while dist_y>10^-3
    global Z, dist_y,r,A,E,w,K,N,policy_a, policy_c, policy_l
    @time λ,r,w, policy_a, policy_c, policy_l,K,N = ayiagary(A,E,r,w,τy,T,β,η,μ,Z)
    Y = Z*K^θ*N^(1-θ)
    dist_y = abs(Y-1)
    Z =4/5*Z + 1/5 * 1/(K^θ*N^(1-θ))
    println("Y is $Y, new Z is $Z")
end


@time λ,r,w, policy_a, policy_c, policy_l,K,N = ayiagary(A,E,r,w,τy,T,β,η,μ,Z)

plot(A,[policy_c[:,3],policy_c[:,5]])
plot(A,[A,A[policy_a[:,1]],A[policy_a[:,4]],A[policy_a[:,5]]], labels = ["45 degrees" "Low e" "Medium e" "High e"])
1/β-1-r

λ1 = zeros(nA,nE)

i=1
for a=1:nA
    global i
    for e=1:nE
        λ1[a,e] = λ[i]
        i+=1
    end
end


plot(A, sum(λ1,dims=2) )



#cd("C:\\Users\\santo279\\Desktop\\HW3_backup")
@save "variables.jld2"  λ,r,w, policy_a, policy_c, policy_l


#k = ayiagary_supply(A,E,r0,w0,100)

#r_range=range(0.0,stop = 1.0,length = 100)
#K = ((r_range.+δ)./θ).^(1/(θ-1))

#@save "SupplyDemand.jld2"  r_range,K,k

#plot(r_range,[K,k])
cd("\\\\tsclient\\C\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
@save "variables.jld2"  λ,r,w, policy_a, policy_c, policy_l
#@save "SupplyDemand.jld2"  r_range,K,k
