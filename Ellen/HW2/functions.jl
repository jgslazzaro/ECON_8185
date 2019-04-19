using NLsolve, LinearAlgebra, ForwardDiff

#This function writes the system as a Sta Space represention

function State_Space(params_calibrated,steadystates, P,Q)

    #= δ  depreciation rate
    θ   #capital share of output
    β   #Discouting
    σ   #Elasticity of Intertemporal Substitution
    ψ     #Labor parameter
    γn    #Population growth rate
    γz   #Productivitu growth rate
    gss  #average g
    τxss  #average τx
    τhss  #average τh
    zss  #average z (z is in logs) =#

    δ,θ,β,σ,ψ,γn,γz = params_calibrated
    gss,τxss,τhss,zss = steadystates

    if ψ == 0
        ψ =eps()
    end

    #Function with the FOCs
    #Note that g and z are nonegative and are defined in logs
    zss = exp(zss)
    gss = exp(gss)

    function SS!(eq,vector::Vector)
        k,h, c= vector
        eq[1]=k/h-((1+τxss)*(1-β*(1+γz)^(-σ)*(1-δ))/(β*(1+γz)^(-σ)*θ*zss^(1-θ)) )^(1/(θ-1))
        eq[2]=c-( (k/h)^(θ-1)*zss^(1-θ) -(1+γz)*(1+γn)+1-δ)*k+gss
        eq[3]=ψ*c-( (1-τhss)*(1-θ)*(k/h)^θ *zss^(1-θ))*(1-h)
    end

    SteadyState = nlsolve(SS!, [3,0.25,.4],ftol = :1.0e-20)
    kss,hss,css = SteadyState.zero
    #GDP
    yss = kss^(θ)*(zss*hss)^(1-θ)
    xss = (1+γz)*(1+γn)*kss-(1-δ)*kss


    #log deviations
    T=numgradient(loglineq1,[kss,kss,hss,zss,τhss,gss])
    a =[-kss*T[1]/(kss*T[1]),-kss*T[2]/(kss*T[1]),-hss*T[3]/(kss*T[1]),
    -zss*T[4]/(kss*T[1]),-τhss*T[5]/(kss*T[1]),-gss*T[6]/(kss*T[1])]
    #if ψ==0
    #    a[1],a[2:end]=-1,zeros(5)
    #end

    T=numgradient(loglineq2,[kss,kss,kss,hss,hss,zss,τxss,gss,zss,τxss,gss])
    b = [kss*T[1]/(-kss*T[1]),kss*T[2]/(-kss*T[1]),kss*T[3]/(-kss*T[1]),hss*T[4]/(-kss*T[1]),
    hss*T[5]/(-kss*T[1]),zss*T[6]/(-kss*T[1]),τxss*T[7]/(-kss*T[1]),gss*T[8]/(-kss*T[1]),
    zss*T[9]/(-kss*T[1]),τxss*T[10]/(-kss*T[1]),gss*T[11]/(-kss*T[1])]

    A1 = [1 0 0; 0 0 0; 0 b[3] b[5]]
    A2 = [0 -1 0; a[1] a[2] a[3]; b[1] b[2] b[4]]
    U = [0 0 0 0 0 0 0 0;
    a[4] a[5] 0 a[6] 0 0 0 0;
    b[6] 0 b[7] b[8] b[9] 0 b[10] b[11]]

    A1,A2
    eig = eigen(A1,-A2)
    V=eig.vectors
    Π = eig.values
    #Sorting
    for j=1:3
    for i=1:2
        if eps(Float64)<abs(Π[i+1])<abs(Π[i])
            Π[i],Π[i+1] = Π[i+1],Π[i]
            V[:,i],V[:,i+1] = V[:,i+1],V[:,i]
        elseif abs(Π[i]) < eps(Float64)
            Π[i],Π[end] =Π[end],Π[i]
            V[:,i],V[:,end]=V[:,end],V[:,i]
        end
    end
    end
    if abs(Π[1])>1
        error("All Eigen Values outside unit circle")
    end
    Π= Diagonal(Π)
    iszero(Π[1])
    #If want to check if these matrics conform (they are equal but there is some roundoff error):
    #A1*V
    #-A2*V*Π

    #CHECK this, inv in the last V or not?
    A = V[1,1]*Π[1,1]*inv(V[1,1])
    C = V[2:end,1]*(V[1,1])
    C = hcat(C,zeros(2,1))

    #=Checking if k coefficients are 0:

    check1 = a[1] + a[2]*A + a[3]*C[2]
    check2 = b[1] + b[2]*A + b[3]*A^2 + b[4]*C[2] + b[5]*C[2]*A

    if abs(check1) >1e-5 || check2 > 1e-5
        println("k coefficients are not 0")
    end
    =#


    #Implementing the code solve the system commented out above as a system of linear equations.
    TOP = hcat(a[2]*Matrix{Float64}(I,4,4),   a[3]*Matrix{Float64}(I,4,4))
    #Everything multiplying B concatenated with stuff multiplying D in the first equations
    BOTTOMLEFT =  b[2]*I + b[3].*A *I + b[3]*P + b[5].*C[2]*I + b[5]*P
    #Everything multiplying B in the last equations
    BOTTOM = hcat(BOTTOMLEFT',  b[4]*Matrix{Float64}(I,4,4)) #Concatenates with stuff multiplying D
    RHS = - vcat([a[4] a[5] 0 a[6]]',  ([b[6] 0 b[7] b[8]].+[b[9] 0 b[10] b[11] ]*P)') #Constant terms
    #Solving the system
    BD = (vcat(TOP,BOTTOM)\RHS)[:]
    D=ones(2,4)
    D[1,:]= BD[1:4]
    D[2,:]= BD[5:8]


    #Rewritting to match Anmol's notation
    A = hcat(vcat(C[1],zeros(4,1)),vcat(D[1,:]',P))
    B = hcat(zeros(5,1),vcat(zeros(1,4),Q))


    #We have h as function of states. To find the Matrix B, we need to find y and x
    #as a function of states

    function kt1(vector::Vector)
        k,z,τh,τx,g = vector
        tilde = log.([k,z,τh,τx,g]).-log.([kss,zss,τhss,τxss,gss])
        for i = 1:length(tilde)
            if isnan(tilde[i])
                tilde[i] = 0
            end
        end

        k1= A[1,:]' * tilde
        return k1
    end


    function ht(vector::Vector)
        k,z,τh,τx,g = vector
        tilde = log.([k,z,τh,τx,g]).-log.([kss,zss,τhss,τxss,gss])
        for i = 1:length(tilde)
            if isnan(tilde[i])
                tilde[i] = 0
            end
        end
        h = C[2,1]*(log(k)-log(kss)) + D[2,:]' * tilde[2:end]
        return h
    end



    function yt(vector::Vector)
        k,z,τh,τx,g = vector
        h = exp(ht(vector)+log(hss))
        y = k^θ * (z*h)^(1-θ)
        return y
    end

    T=numgradient(yt,[kss,zss,τhss,τxss,gss])
    ycoefs = [kss*T[1]/yss,zss*T[2]/yss,τhss*T[3]/yss,τxss*T[4]/yss,gss*T[5]/yss]

    yt([kss,zss,τhss,τxss,gss])

    function xt(vector::Vector)
        k,z,τh,τx,g = vector
        k1 = exp(kt1(vector)+log(kss))
        x= (1+γn)*(1+γz)k1 - (1-δ)k

        return x
    end

    T=numgradient(xt,[kss,zss,τhss,τxss,gss])
    xcoefs = [kss*T[1]/xss,zss*T[2]/xss,τhss*T[3]/xss,τxss*T[4]/xss,gss*T[5]/xss]

    C = [ycoefs[1] ycoefs[2] ycoefs[3] ycoefs[4] ycoefs[5];
    xcoefs[1] xcoefs[2] xcoefs[3] xcoefs[4] xcoefs[5];
    C[2,1] D[2,1] D[2,2] D[2,3] D[2,4];
    0 0 0 0 1]

return A,B,C
end


#Find the numerical gradient of a function using forweard numerical differentiation
function numgradient(f,arguments::Vector;step=1e-10)
    n=length(arguments)
    grad = zeros(n)
    for i=1:n
        e = zeros(n)
        e[i] = 1
        grad[i] = (f(arguments.+e*step)-f(arguments))/step
    end
    return grad
end

#Euler equations to be log linearized:

function loglineq1(vector::Vector;params_calibrated =params_calibrated)
    k,k1,h,z,τh,g= vector
    δ,θ,β,σ,ψ,γn,γz = params_calibrated
    c = k^θ  * ((z *h)^(1-θ)) - ((1+γz)*(1+γn)*k1-(1-δ)*k+ g )
    eq =(ψ *c)^(1/θ)  - (k/h)*((1-h)*(1-τh)*(1-θ)*z^(1-θ))^(1/θ)

    return eq
end
function loglineq2(vector::Vector; params_calibrated =params_calibrated)
    k,k1,k2,h,h1,z,τx,g,z1,τx1,g1 = (vector)
    δ,θ,β,σ,ψ,γn,γz = params_calibrated
    c = k^θ  * ((z *h)^(1-θ)) - ((1+γz)*(1+γn)*k1-(1-δ)*k+ g )
    c1 = k1^θ * ((z1 *h1)^(1-θ)) - ((1+γz)*(1+γn)*k2-(1-δ)*k1+ g1 )
    eq =  (c^(-σ) *(1-h)^(ψ*(1-σ))*(1+τx)  - (1-δ)*(1+τx1)* β*(1+γz)^(-σ) * c1^(-σ) * (1-h1)^(ψ*(1-σ)))^(-1/θ) -
       (β*(1+γz)^(-σ) * c1^(-σ) * (1-h1)^(ψ*(1-σ)) * θ*(z1*h1)^(1-θ))^(-1/θ)* k1
    return eq
end


#Utility function
#REMEMBER: V is the [X u]' Vector, hence [k0 z0 τh0 τx0 g0 1 K0 H0 k1 h0]'
#X is the state variables vector (and a constant) while u is the cotrol variables vector
function u(x::Vector;params_calibrated = params_calibrated)
    δ,θ,β,σ,ψ,γn,γz = params_calibrated
    k, z, τh, τx, g,cons,K,H, k1, h = x
    r = θ*K^(θ-1)*(exp(z)*H)^(1-θ)
    w = (1-θ)*K^θ*H^(-θ) * exp(z)^(1-θ)
    c = r*k + (1-τh)*w*h + g -(1+τx)*((1+γn)*(1+γz)*k1-(1-δ)*k)
    if c<=0
        u = -Inf #any value of negative consumtpion in the grid will be avoided
    elseif  (σ ==1.0 && ψ!=0)
        u = log(c) + ψ * log(1-h)
    elseif (σ ==1.0 && ψ==0)
        u = log(c)
    else
        u = (c*(1-h)^ψ)^(1-σ) / (1-σ)

    end
    return u
end

function LQ_distorted(params_calibrated,steadystates)
    #This function implements the LQ method for a distorted economy
#Function with the FOCs
#Note that g and z are nonegative and are defined in logs

δ,θ,β,σ,ψ,γn,γz = params_calibrated
gss,τxss,τhss,zss = steadystates

if ψ == 0
    ψ =eps()
end

#Function with the FOCs
#Note that g and z are nonegative and are defined in logs
zss = exp(zss)
gss = exp(gss)
function SS!(eq,vector::Vector)
    k,h, c= vector
    if σ!=1
    eq[1]=k/h-((1+τxss)*(1-β*(1+γz)^(-σ)*(1-δ))/(β*(1+γz)^(-σ)*θ*zss^(1-θ)) )^(1/(θ-1))
    eq[2]=c-( (k/h)^(θ-1)*zss^(1-θ) -(1+γz)*(1+γn)+1-δ)*k+gss
    eq[3]=ψ*c-( (1-τhss)*(1-θ)*(k/h)^θ *zss^(1-θ))*(1-h)
    else
    eq[1] = ((1+τxss)*((1+γz)*(1+γn)/β  -(1-δ)))^(1/(1-θ))*k -(θ)^(1/(1-θ))*zss*h
    eq[2] = (ψ*c*zss^(θ-1)/((1-τhss)*(1-θ)))^(1/θ) - k/h *(1-h)^(1/θ)
    eq[3] = c -( θ*k^θ*(zss*h)^(1-θ)+(1-τxss)*(1-θ)*k^θ*(zss*h)^(1-θ) -(1+τxss)*((1+γn)*(1+γz)k-(1-δ)*k))
    end
end



SteadyState = nlsolve(SS!, [3,0.25,4],ftol = :1.0e-20)
kss,hss,css = SteadyState.zero
#GDP
yss = kss^(θ)*(zss*hss)^(1-θ)
xss = (1+γz)*(1+γn)*kss-(1-δ)*kss
wss = (1-θ)*kss^θ*hss^(-θ) * exp(zss)^(1-θ)
rss = θ*kss^(θ-1)*(exp(zss)*hss)^(1-θ)

#vector with SS variables And the consant term
vss= [kss,zss,τhss, τxss, gss,1,kss,hss, kss, hss]
#Find the Gradient and Hessian AT THE SS
uss=u(vss)

∇u = ForwardDiff.gradient(u,vss)
Hu = ForwardDiff.hessian(u,vss)


#We are looking for a Linear quadratic matrix M such that u(vss) = vss'M vss
#Applying Sargent's Formula for 2nd order Taylor Expansion:
e = [0;0;0;0;0;1;0;0;0;0] #thus e'v=1

M = e.* (uss - ∇u'*vss .+ (0.5 .* vss' *Hu *vss) ) * e' +
    0.5 *(∇u * e' - e * vss' * Hu - Hu*vss*e' +e* ∇u') +
    0.5 * Hu


vss'*M*vss



#Translating M into the Matrices we need:
Q = M[1:8,1:8]
W = M[1:8,9:10]
R = M[9:10,9:10]

xss = vss[1:8]
uss = vss[9:10]

xss'*Q*xss+uss'*R*uss+2*xss'*W*uss



A = [0 0 0 0 0 0 0 0; #capital
    0 ρz ρzh ρzx ρzg 0 0 0; #z
    0 ρhz ρh ρhx ρhg 0 0 0; #τh
    0 ρxz ρxh ρx ρxg 0 0 0; #τx
    0 ρgz ρgh ρgx ρg 0 0 0; #g
    0 0 0 0 0 1 0 0; #Constant
    0 0 0 0 0 0 0 0; #K To be determined
    0 0 0 0 0 0 0 0] #H To be determined]

B =[1 0;
    0 0;
    0 0 ;
    0 0 ;
    0 0 ;
    0 0 ;
    0 0 ;
    0 0 ]

C = [0 0 0 0;
    σz σzh σzx σzg;
    σzh σh σhx σhg ;
    σzx σhx σx σxg ;
    σzg σhg σxg σg;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0]

    #Mapping to the problem without discounting (1 VARIABLES ARE ~ IN LECTURE NOTES)
A1 = sqrt(β) *(A -B* (R\W'))
B1 = sqrt(β) *B
Q1 = Q- W*(R\W')


xss1 = xss*β^(1/2)
uss1 = (uss+R\W'*xss)*β^(1/2)

xss1'*Q1*xss1 + uss1'*R*uss1


A1y = A1[1:6,1:6]
A1z = A1[1:6,7:8]
B1y = B1[1:6,:]
Q1y = Q1[1:6,1:6]
Q1z = Q1[1:6,7:8]

#Big K litlle k:
Θ = [1 0 0 0 0 0;
    0 0 0 0 0 0]
Ψ = [0 0;
    0 1]
Wz = W[7:8,:]
Wy = W[1:6,:]
Θ1 = (I+ Ψ*(R\Wz')) \ (Θ - Ψ*(R\Wy'))
Ψ1 = (I+ Ψ*(R\Wz')) \ Ψ

#2 Variables are ^ in lecture NOTES

A2 = A1y + A1z*Θ1
B2 = B1y + A1z*Ψ1
Q2 = Q1y+Q1z*Θ1
Abar = A1y - B1y*(R\Ψ1')*Q1z'

P,F1 = Vaughan(A2,B2,R,B1y,Q2,Abar)


return R, B1y,P,F1,B2,A2[1:6,1:6],kss,hss,Wy
end

function Vaughan(A2,B2,R,B1y,Q2,Abar)
    L=size(A2)[1]
    ℋ = [inv(A2)  (A2\B2)*(R\B1y');Q2/A2 Q2*(A2\B2)*(R\B1y')+Abar'] #This is the coefficient matrix.


    eigs= eigen(ℋ) #Take the eigenvector matrix The first
    #Sorting Eigenvalues and eigenvectors
    Λ = sort(eigs.values,by=abs,rev = true)
    V =zeros(2*L,2*L)
    for i=1:2*L
        V[:,i] = eigs.vectors[:,findfirst((eigs.values).==(Λ[i]))]
    end

    P = (V[L+1:end,1:L]) / (V[1:L,1:L])  #Get the P matrix
    F1 = (R+B1y'*P*B2)\B1y'*P*A2
    #F2= (R+B2'*P*B2) \ B2' * P * A2
    #F = F2 + R\Wy' #Finally, compute F
    return P,F1
end
