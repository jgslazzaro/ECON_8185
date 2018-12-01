
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
#cd("\\\\tsclient\\C\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
include("functions.jl")
using  LinearAlgebra, Plots
using  JLD2,FileIO


#Defining Parameters:
β = 0.98 #Discount rate
μ = 1.5  #Elasticity of intertemporal substitution
η = 0.3 #Utility parameter
τy = 0.2 #Income tax
ρ = 0.6 #autocorrelation
σ = 0.3 #Variance
δ = 0.075 #Depreciation rate
θ = 0.3 #Capital Share
T=0.1 #Transfers
G = 0.2 #Expenses Government
Z=19.12048323273413#productivity level (not used)

fast = false #Used to get the policy functions faster, assuming we know the equilibrium r and w

amin = -0.0   #Debt limit
amax= 13.0 #capital limit

nE = 5 #Number of states for e
nA = 500#states for assets

r= (1/β - 1)-0.0001 #initial guess for r liens in the interval (-δ, 1/β-1)

Kguess = ((r+δ)/(Z*θ))^(1/(θ-1))#K for the nitial guess of r
w=  (1-θ)*Z*Kguess^θ  #Initial wage given r0
Nguess = (w/(Z*(1-θ)))^(-1/θ)*Kguess
##

#Defining grids

pdfE,E = Tauchen(ρ,σ,nE)    #E comes from Tauchen method
A = range(amin,stop = amax, length = nA) #Half points will be in the first third of the grid

#Calibrating Z Please use this with small grid
#=dist_y = 1
while dist_y>10^-3
    global Z, dist_y,r,Assets,E,w,K,N,policy_a, policy_c, policy_l,Y
    @time λ,r,w, policy_a, policy_c, policy_l,K,N,Y,B,K = ayiagary(A,E,r,w,τy,T,β,η,μ,Z,G)
    dist_y = abs(Y-1)
    Z =1/2*Z+1/2 * Z/Y
    println("Y is $Y, new Z is $Z")
end=#


λ,r,w, policy_a, policy_c, policy_l,Assets,N,Y,B,K = ayiagary(A,E,r,w,τy,T,β,η,μ,Z,G,fast)
Y
plot(A,[policy_c[:,3],policy_c[:,5]])
plot(A,[A,A[policy_a[:,1]],A[policy_a[:,4]],A[policy_a[:,5]]],legend = :bottomright ,
labels = ["45 degrees" "Low e" "Medium e" "High e"])
maximum(policy_a)

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


F = zeros(nA+1)
S = zeros(nA+1)
mean_wealth = 0
gini = 0

for e=2:nA+1
    global F,S,mean_wealth,gini
        F[e] = F[e-1] + sum(λ1[e-1,:])
        S[e] = S[e-1] + sum(λ1[e-1,:])*A[e-1]
        mean_wealth += A[e-1]*sum(λ1[e-1,:])

        for e1 = 1:nA
            gini += sum(λ1[e-1,:])*sum(λ1[e1,:])*abs(A[e-1]-A[e1])
        end
end
gini = 1/(2*mean_wealth) * gini
L = S./S[end]
plot(F,[L,F],legend = false, xaxis="share of population (ordered by wealth)",yaxis = "Share of wealth owned")

var_wealth = 0
for a=1:nA
    global var_wealth
    var_wealth += sum(λ1[a,:])*(A[a]-mean_wealth)^2
end

std_wealth = sqrt(var_wealth)

#cd("C:\\Users\\santo279\\Desktop\\HW3_backup")
@save "40.jld2"


#k = ayiagary_supply(A,E,r0,w0,100)

#r_range=range(0.0,stop = 1.0,length = 100)
#K = ((r_range.+δ)./θ).^(1/(θ-1))

#@save "SupplyDemand.jld2"  r_range,K,k

#plot(r_range,[K,k])
cd("\\\\tsclient\\C\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW3")
@save "variables.jld2"  λ,r,w, policy_a, policy_c, policy_l
#@save "SupplyDemand.jld2"  r_range,K,k
