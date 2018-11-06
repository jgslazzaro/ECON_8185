using Plots, NLsolve, ForwardDiff, DataFrames
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW2")

#Parameters:
δ = 0.05    #depreciation rate
θ = 1/3  #capital share of output
β = 0.975  #Discouting
σ = 1  #Elasticity of Intertemporal Substitution
ψ = 2    #Labor parameter
γn= 0.003    #Population growth rate
γz= 0.004   #Productivitu growth rate
gss = 0.03 #average g
τxss = 0.01 #average τx
τhss = 0.02 #average τh
zss = 1 #average z

#Function with the FOCs
function SS!(eq, vector::Vector)
    k,h = (vector)
    k1 = k
    h1 = h
    g, τx,τh, z = gss,τxss,τhss, zss
    z1 = z
    τx1 = τx
    c = k^θ * (z *h)^(1-θ) - ((1+γz)*(1+γn)*k1-(1-δ)*k+g)
    c1 = c
    eq[1] = ψ *c/(1-h) - (1-τh)*(1-θ) *(k/h)^θ *z^(1-θ)
    eq[2] = c^(-σ) *(1-h)^(ψ*(1-σ))*(1+τx)  -  β*(1+γz)^(-σ) * c1^(-σ) * (1-h1)^(ψ*(1-σ)) *(θ*k1^(-θ)*(z1*h1)^(1-θ)+(1-δ)*(1+τx1))
    return eq
end

SS!(eq,[1.0,3])
SteadyState = nlsolve(SS!, [0.05,0.97],ftol = :1.0e-20, method = :trust_region , autoscale = true)
kss,hss = SteadyState.zero
SS!([1,1.1],[kss,hss])

function loglineq1(vector::Vector)
    k,k1,h,z,τh,g= (vector)

    c = k^θ * (z *h)^(1-θ) - ((1+γz)*(1+γn)*k1-(1-δ)*k+g)
    eq = ψ *c/(1-h) - (1-τh)*(1-θ) *(k/h)^θ *z^(1-θ)

    return eq
end
function loglineq2(vector::Vector)
    k,k1,k2,h,h1,z,τx,g,z1,τx1,g1 = (vector)
    c = k^θ * (z *h)^(1-θ) - ((1+γz)*(1+γn)*k1-(1-δ)*k+g)
    c1 = k1^θ * (z1 *h1)^(1-θ) - ((1+γz)*(1+γn)*k2-(1-δ)*k1+g1)
    eq = c^(-σ) *(1-h)^(ψ*(1-σ))*(1+τx)  -  β*(1+γz)^(-σ) * c1^(-σ) * (1-h1)^(ψ*(1-σ)) *(θ*k1^(-θ)*(z1*h1)^(1-θ)+(1-δ)*(1+τx1))
    return eq
end


#log deviations
T=ForwardDiff.gradient(loglineq1,[kss,kss,hss,zss,τhss,gss])
A =[-kss*T[1]/(kss*T[1]),-kss*T[2]/(kss*T[1]),-hss*T[3]/(kss*T[1]),
-zss*T[4]/(kss*T[1]),-τhss*T[5]/(kss*T[1]),-gss*T[6]/(kss*T[1])]

T=ForwardDiff.gradient(loglineq2,[kss,kss,kss,hss,hss,zss,τxss,gss,zss,τxss,gss])
B = [kss*T[1]/(-kss*T[1]),kss*T[2]/(-kss*T[1]),kss*T[3]/(-kss*T[1]),hss*T[4]/(-kss*T[1]),
hss*T[5]/(-kss*T[1]),zss*T[6]/(-kss*T[1]),τxss*T[7]/(-kss*T[1]),gss*T[8]/(-kss*T[1]),
zss*T[9]/(-kss*T[1]),τxss*T[10]/(-kss*T[1]),gss*T[11]/(-kss*T[1])]

A1 = [1 0 0; 0 0 0; 0 B[3] B[5]]
A2 = [0 -1 0; A[1] A[2] A[3]; B[1] B[2] B[4]]
U = [0 0 0 0 0 0 0 0;
A[4] A[5] 0 A[6] 0 0 0 0;
B[6] 0 B[7] B[8] B[9] 0 B[10] B[11]]

chur = eigen(A1,-A2,[false,true,false])
V=chur.vectors
Π = Diagonal(chur.values)

V11=V[:,2]
Π11=Π[2,2]
V21 =ones(3,2)
V21[:,1]=V[:,1]
V21[:,2] = V[:,3]

A = V11'*Π11 *V11
