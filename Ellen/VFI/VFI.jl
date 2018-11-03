#This code defines the VFI function as well its auxiliary functions, notably Tauchen.

#Consumption with non-negative investment option
#Consumption with non-negative investment option
function consumption(k,z,h,ϕ=0,δ = 1,θ=1/3, InvestmentNonNegative = false)
    K=length(k)
    Z=length(z)
    H=length(h)
    #Preallocating memory. Note loops are fast in Julia.
    c=Array{Float64,4}(undef,K,Z,K,H)
    if InvestmentNonNegative == true #Non Negative investment
        for k0 in 1:K, k1 in 1:K
            x = k[k1]-(1-δ)*k[k0] #define the investment variable
            if x>=0     #If it is positive, we assign a normal value for consumption
                for z0 in 1:Z , h0 in 1:H
                    c[k0,z0,k1,h0]=k[k0]^θ *(z[z0]*h[h0])^(1-θ) + (1-δ)k[k0] - k[k1]
                end
            else
                c[k0,:,k1,:] = -ones(1,Z,1,H) #If not, consumption is negative and will
                                      #be ruled out by Utility function
            end
        end
    else #Standard case: investment can be negative
        for k0 in 1:K, z0 in 1:Z, k1 in 1:K, h0 in 1:H
            c[k0,z0,k1,h0]=k[k0]^θ *z[z0]^(1-θ) + (1-δ)k[k0] - k[k1]
        end
    end
    return c
end


function log_utility(c)
    if c<=0
        u = -Inf #any value of negative consumtpion in the grid will be avoided
    else
        u = log(c)
    end
    return u
end


using Distributions
function Tauchen(ρ,σ,Y,μ = 0,m = 3)
    #This function is to discretize an AR(1) process following Tauchen(1986) method
    # y_{t+1} = μ + ρy_t + ϵ
    #ϵ~N(0,σ^2)
    #Y is the number of y states
if Y>1
    ybar = μ/(1-ρ)
    ymax= ybar + m*(σ^2/(1-ρ^2))^(1\2) #maximum y
    ymin= ybar - m*(σ^2/(1-ρ^2))^(1\2) #minimum y

    Δ = (ymax-ymin)/(Y-1)# #distance between each y
    y=ymin:Δ:ymax #vector of possible states of p

    d=Normal()

    pdfY=ones(Y,Y) #preallocate memory and create the transition matrix in the following loop
    for i in 1:Y
        pdfY[i,1]=cdf(d,(y[1] + Δ/2 -ρ*y[i]) / σ^0.5);
        pdfY[i,Y]=1-cdf(d,(y[Y] -Δ/2 - ρ*y[i]) / σ^0.5);
        for j in 2:Y-1
            pdfY[i,j]=cdf(d,(y[j] + Δ/2 - ρ*y[i])/σ^0.5) - cdf(d,(y[j] - Δ/2 - ρ*y[i]) / σ^0.5);
        end
    end
else
    y=μ
    pdfY=1
end

    return pdfY, y
end


function VFI(δ,θ,β,ρ,σ,μ,ϕ=0,K=500,kmax=10,kmin=0.01,Z=10,Xnneg=false,tol=10^(-8))
#parameters:
#δ - depreciation rate
#θ - capital share of output
#β - Discouting
#ρ - AR coefficient
#σ - AR shock SD
#ϕ - Labor parameter
#μ - AR(1) constant term
#K - number of gridpoints for capital AND LABOR
#Xnneg - Non negative investment (true or false)
#Z - number of productivity states
#kmax - maximum value for k (set low if no productivity shocks!)
#kmin - minimum value for k
#tol - convergence parameter
#grid for k
    k = range(kmin,stop = kmax, length=K)

#grid for z
    Π, z = Tauchen(ρ,σ,Z,μ) #Transition matrix and values for log(z)
    z=exp.(z) #recover z

#utility Function:
if ϕ>0 #with labor
    h = range(0,stop = 1, length = K)
    #labu is the utility for satte value and control value
    labu = log_utility.(consumption(k,z,h,ϕ,δ,θ, Xnneg))

    #u is the utility after solving he static labor problem
    u = Array{Float64,3}(undef,K,Z,K)
    policy_h1 = Array{Int64,3}(undef,K,Z,K)
    for k0 in 1:K, z0 in 1:Z, k1 in 1:K
        u[k0,z0,k1], policy_h1[k0,z0,k1] = findmax(labu[k0,z0,k1,:] .+ ϕ.*log.(1 .- h))
    end
else #without labor
    h = 1
    u = log_utility.(consumption(k,z,h,ϕ,δ,θ, Xnneg)[:,:,:,1])
    policy_h = ones(K,Z)
end
#utility values values, with or without nonnnegativity constraint on investment
#Note: Consumption and labu dimension is (K,Z,K,K)
#so they can potentially take a huge space in memory.
#Now we may forget about labor until we have the policy function for K

#Initial value function guess:
    V  = 0 .* Array{Float64,2}(undef,K,Z)
    Vf = 0 .* Array{Float64,2}(undef,K,Z)

    #preallocation
    policy_k = Array{Int64,2}(undef,K,Z)
    #loop stuff
    distance = 1

#iterating on Value functions:
while distance >= tol
    distance, Vf, policy_k
    Vf = copy(V) #save V to compare later

    #find the expected Vf: E[V(k,z')|z] = π(z1|z)*V(k,z1) +π(z2|z)*V(k,z2)
    EVf= (Π*Vf')'  #The transposing fix the dimensions so it works!

    for k0 in 1:K, z0 in 1:Z
        V[k0,z0] , policy_k[k0,z0] = findmax(u[k0,z0,:] .+ β * EVf[:,z0]) #Bellman Equation
    end
 distance = maximum(abs.(V-Vf))

end
#Finally, find labor policy:
    if ϕ>0
        policy_h = Array{Int64,2}(undef,K,Z)
        for k0 in 1:K, z0 in 1:Z
            policy_h[k0,z0] = policy_h1[k0,z0,policy_k[k0,z0]]
        end
    end

    return V, policy_k,policy_h, k,h, z, Π
end
