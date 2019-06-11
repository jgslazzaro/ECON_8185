#This is the main file used to generate pictures for HW1

#First Calibration, full depreciation and exogenous labor:

δ = 1    #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
ρ = 0.5  #AR coefficient
σ = 0.5  #AR shock SD
μ = 0    #AR(1) constant term
ϕ = 0    #Labor parameter
γn= 0.    #Population growth rate4
γz= 0    #Productivitu growth rate
β = β*(1+γn)

Xnneg = true #Non negative investment

include("VFI.jl")



include("Vaughan.jl")
include("Riccati.jl")
@time A,B,C,P ,F,A1,B1 ,F1,kss,hss = LQ(δ,β,ρ,σ,μ,ϕ,γn,γz) #LQ objects

@time Av,Bv,Cv,Fv,Pv,kssv,hssv = run_Vaughan(δ,β,ρ,σ,μ,ϕ,γn,γz)


#VFI
#grid for k
K = 60 #number of gridpoints for capital
kmax = 3*kss # maximum value for k (set low if no productivity shocks!)
kmin = 1/5*kss #minimum value for k
#grid for z
Z = 40 #number of productivity states

@time VFI_V, VFI_policyk,VFI_policyh, k,h, z, Π = VFI(δ,θ,β,ρ,σ,μ,γn,γz,ϕ,K,kmax,kmin,Z,Xnneg)



LQ_policyk = -F[1,1].*k .+ -F[1,2] * (0) .+ -F[1,3]
LQ_policyh = collect(-F[2,1].*k .+ -F[2,2] * (0) .+ -F[2,3])
Vaughan_policyk = -Fv[1,1].*k .+ -Fv[1,2] * (0) .+ -Fv[1,3]
Vaughan_policyh = collect(-Fv[2,1].*k .+ -Fv[2,2] * (0) .+ -Fv[2,3])



using Plots
figures = Array{Plots.Plot{Plots.GRBackend}}(undef, 8)

title = "Capital Policy Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz) "
figures[1] = plot(k,[k,θ*β*z[20]*k.^(θ),k[VFI_policyk[:,20]],LQ_policyk,Vaughan_policyk],
label = ["45","True","VFI","LQ","Vaughan"],legend=:topleft,title = title)

title = "Labor Policy Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz) "
figures[2] = plot(k,[ones(K),h[VFI_policyh[:,20]],LQ_policyh,Vaughan_policyh],
label = ["True","VFI","LQ","Vaughan"],legend=:bottomleft,title = title)




#=Value Functions
LQ_VF = zeros(K)
Vaughan_VF = zeros(K)
#P[3,3] = P[3,3]-0.01
for i = 1:K
    X =[k[i],0,1]
    LQ_VF[i] = X'*P*X
    Vaughan_VF[i] = X'*Pv*X
end
trueVF = θ/(1-θ*β) .* log.(k) .+ 1/(1-β) *((θ*β)/(1-θ*β) *log(θ*β) + log(1-θ*β))

title = "Value Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz) "
figures[2] = plot(k,[trueVF,VFI_V[:,20],LQ_VF,Vaughan_VF],
label = ["True","VFI","LQ","Vaughan"],legend=:bottomright,title = title)=#


#Second Calibration, Depreciation is not full:

δ = 0.05    #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
ρ = 0.5  #AR coefficient
σ = 0.5  #AR shock SD
μ = 0    #AR(1) constant term
ϕ = 0    #Labor parameter
γn= 0   #Population growth rate
γz= 0    #Productivitu growth rate
β = β*(1+γn)
Xnneg = true #Non negative investment

@time A,B,C,P ,F,A1,B1 ,F1,kss,hss = LQ(δ,β,ρ,σ,μ,ϕ,γn,γz) #LQ objects
@time Av,Bv,Cv,Fv,Pv,kssv,hssv = run_Vaughan(δ,β,ρ,σ,μ,ϕ,γn,γz)



#VFI
#grid for k
kmax = 3*kss # maximum value for k (set low if no productivity shocks!)
kmin = 1/5*kss #minimum value for k


@time VFI_V, VFI_policyk,VFI_policyh, k,h, z, Π = VFI(δ,θ,β,ρ,σ,μ,γn,γz,ϕ,K,kmax,kmin,Z,Xnneg)


LQ_policyk = -F[1,1].*k .+ -F[1,2] * (0) .+ -F[1,3]
LQ_policyh = collect(-F[2,1].*k .+ -F[2,2] * (0) .+ -F[2,3])
Vaughan_policyk = -Fv[1,1].*k .+ -Fv[1,2] * (0) .+ -Fv[1,3]
Vaughan_policyh = collect(-Fv[2,1].*k .+ -Fv[2,2] * (0) .+ -Fv[2,3])


title = "Capital Policy Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz)"
figures[3] = (plot(k,[k,k[VFI_policyk[:,20]],LQ_policyk,Vaughan_policyk],
label = ["45","VFI","LQ","Vaughan"],legend=:topleft,title = title))

title = "Labor Policy Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz) "
figures[4] = plot(k,[h[VFI_policyh[:,20]],LQ_policyh,Vaughan_policyh],
label = ["VFI","LQ","Vaughan"],legend=:bottomleft,title = title)


#Third Calibration, Depreciation is not full, technology growth:

δ = 0.05    #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
ρ = 0.5  #AR coefficient
σ = 0.5  #AR shock SD
μ = 0    #AR(1) constant term
ϕ = 0    #Labor parameter
γn= 0   #Population growth rate
γz= 0.05    #Productivitu growth rate
β = β*(1+γn)
Xnneg = true #Non negative investment

@time A,B,C,P ,F,A1,B1 ,F1,kss,hss = LQ(δ,β,ρ,σ,μ,ϕ,γn,γz) #LQ objects
@time Av,Bv,Cv,Fv,Pv,kssv,hssv = run_Vaughan(δ,β,ρ,σ,μ,ϕ,γn,γz)



#VFI
#grid for k

kmax = 3*kss # maximum value for k (set low if no productivity shocks!)
kmin = 1/5*kss #minimum value for k


@time VFI_V, VFI_policyk,VFI_policyh, k,h, z, Π = VFI(δ,θ,β,ρ,σ,μ,γn,γz,ϕ,K,kmax,kmin,Z,Xnneg)


LQ_policyk = -F[1,1].*k .+ -F[1,2] * (0) .+ -F[1,3]
LQ_policyh = collect(-F[2,1].*k .+ -F[2,2] * (0) .+ -F[2,3])
Vaughan_policyk = -Fv[1,1].*k .+ -Fv[1,2] * (0) .+ -Fv[1,3]
Vaughan_policyh = collect(-Fv[2,1].*k .+ -Fv[2,2] * (0) .+ -Fv[2,3])


title = "Capital Policy Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz)"
figures[5] = (plot(k,[k,k[VFI_policyk[:,20]],LQ_policyk,Vaughan_policyk],
label = ["45","VFI","LQ","Vaughan"],legend=:topleft,title = title))

title = "Labor Policy Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz) "
figures[6] = plot(k,[h[VFI_policyh[:,20]],LQ_policyh,Vaughan_policyh],
label = ["VFI","LQ","Vaughan"],legend=:bottomleft,title = title)



#Fourth Calibration, Depreciation is not full, Population growth:

δ = 0.05    #depreciation rate
θ = 1/3  #capital share of output
β = 0.9  #Discouting
ρ = 0.5  #AR coefficient
σ = 0.5  #AR shock SD
μ = 0    #AR(1) constant term
ϕ = 0.    #Labor parameter
γn= 0.05   #Population growth rate
γz= 0.    #Productivitu growth rate
β = β*(1+γn)
Xnneg = true #Non negative investment

@time A,B,C,P ,F,A1,B1 ,F1,kss,hss = LQ(δ,β,ρ,σ,μ,ϕ,γn,γz) #LQ objects
@time Av,Bv,Cv,Fv,Pv,kssv,hssv = run_Vaughan(δ,β,ρ,σ,μ,ϕ,γn,γz)



#VFI
#grid for k

kmax = 3*kss # maximum value for k (set low if no productivity shocks!)
kmin = 1/5*kss #minimum value for k


@time VFI_V, VFI_policyk,VFI_policyh, k,h, z, Π = VFI(δ,θ,β,ρ,σ,μ,γn,γz,ϕ,K,kmax,kmin,Z,Xnneg)


LQ_policyk = -F[1,1].*k .+ -F[1,2] * (0) .+ -F[1,3]
LQ_policyh = collect(-F[2,1].*k .+ -F[2,2] * (0) .+ -F[2,3])
Vaughan_policyk = -Fv[1,1].*k .+ -Fv[1,2] * (0) .+ -Fv[1,3]
Vaughan_policyh = collect(-Fv[2,1].*k .+ -Fv[2,2] * (0) .+ -Fv[2,3])


title = "Capital Policy Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz)"
figures[7] = (plot(k,[k,k[VFI_policyk[:,20]],LQ_policyk,Vaughan_policyk],
label = ["45","VFI","LQ","Vaughan"],legend=:topleft,title = title))

title = "Labor Policy Function \\delta =$(δ), \\phi=$(ϕ), \\gamma_{n}=$(γn),\\gamma_{z} =$(γz) "
figures[8] = plot(k,[h[VFI_policyh[:,20]],LQ_policyh,Vaughan_policyh],
label = ["VFI","LQ","Vaughan"],legend=:bottomleft,title = title)


#Saving Figures


cd("Figures")

for i=1:8

    savefig(figures[i],"figure$(i).png")
end
