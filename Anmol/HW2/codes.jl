using Plots, NLsolve, ForwardDiff, DataFrames, LinearAlgebra
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

#Parameters to be estimated
ρg = 0.0
ρx = 0.0
ρh = 0.0
ρz = 0.0
σg= 0.00
σx = 0.00
σz = 0.00
σh = 0.00

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

SteadyState = nlsolve(SS!, [0.05,0.97],ftol = :1.0e-20, method = :trust_region , autoscale = true)
kss,hss = SteadyState.zero


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
a =[-kss*T[1]/(kss*T[1]),-kss*T[2]/(kss*T[1]),-hss*T[3]/(kss*T[1]),
-zss*T[4]/(kss*T[1]),-τhss*T[5]/(kss*T[1]),-gss*T[6]/(kss*T[1])]

T=ForwardDiff.gradient(loglineq2,[kss,kss,kss,hss,hss,zss,τxss,gss,zss,τxss,gss])
b = [kss*T[1]/(-kss*T[1]),kss*T[2]/(-kss*T[1]),kss*T[3]/(-kss*T[1]),hss*T[4]/(-kss*T[1]),
hss*T[5]/(-kss*T[1]),zss*T[6]/(-kss*T[1]),τxss*T[7]/(-kss*T[1]),gss*T[8]/(-kss*T[1]),
zss*T[9]/(-kss*T[1]),τxss*T[10]/(-kss*T[1]),gss*T[11]/(-kss*T[1])]

A1 = [1 0 0; 0 0 0; 0 b[3] b[5]]
A2 = [0 -1 0; a[1] a[2] a[3]; a[1] a[2] a[4]]
U = [0 0 0 0 0 0 0 0;
a[4] a[5] 0 a[6] 0 0 0 0;
b[6] 0 b[7] b[8] b[9] 0 b[10] b[11]]

eig = eigen(A1,-A2)
V=eig.vectors
#Sorting
V[:,1], V[:,2] =V[:,2], V[:,1]
Π = Diagonal([eig.values[2], eig.values[1], eig.values[3]])

#If want to check if these matrics conform (they are equal but there is some roundoff error):
A1*V
-A2*V*Π

#CHECK this, inv in the last V or not?
A = V[1,1]*Π[1,1]*inv(V[1,1])
C = V[2:end,1]*(V[1,1])

P = [ρg 0 0 0;
0 ρx 0 0 ;
0 0 ρh 0 ;
0 0 0 ρz]
D = [σg 0 0 0;
0 σx 0 0 ;
0 0 σh 0 ;
0 0 0 σz]

function system!(eq,vector::Vector)
    #vector = rand(8)
    #eq= rand(8)
    B=vector[1:4]'
    D2 = vector[5:8]'

    eq[1:4] = a[2].*B.+a[3].+a[3].*D2.+[a[4] a[5] 0 a[6]]
    eq[5:8] = b[2].*B.+b[3].*A.*B+b[3].*B*P.+b[4].*D2.+b[5].*C[2].+b[5].*B*P.+[b[6] 0 b[7] b[8]].+[b[9] 0 b[10] b[1]]*P
 return     eq
end
D = nlsolve(system!, [1.0,1,1,1,1,1,1,1],ftol = :1.0e-20, method = :trust_region , autoscale = true)
