
#This function writes the system as a Sta Space represention
function State_Space(params_calibrated,steadystates, P,Q;nl=false)

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

    #Function with the FOCs
    zss = exp(zss)

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


    function loglineq1(vector::Vector)
        k,k1,h,z,τh,g= vector

        c = k * ((z *h)^(1-θ))^(1/θ) - ((1+γz)*(1+γn)*k1-(1-δ)*k+ g )^(1/θ)
        eq =(ψ *c)^(1/θ)  - (k/h)*((1-h)*(1-τh)*(1-θ)*z^(1-θ))^(1/θ)

        return eq
    end
    function loglineq2(vector::Vector)
        k,k1,k2,h,h1,z,τx,g,z1,τx1,g1 = (vector)
        c = k * ((z *h)^(1-θ))^(1/θ) - ((1+γz)*(1+γn)*k1-(1-δ)*k+ g )^(1/θ)
        c1 = k * ((z1 *h1)^(1-θ))^(1/θ) - ((1+γz)*(1+γn)*k2-(1-δ)*k1+ g1 )^(1/θ)
        eq =  (c^(-σ) *(1-h)^(ψ*(1-σ))*(1+τx)  - (1-δ)*(1+τx1)* β*(1+γz)^(-σ) * c1^(-σ) * (1-h1)^(ψ*(1-σ)))^(-1/θ) -
         (β*(1+γz)^(-σ) * c1^(-σ) * (1-h1)^(ψ*(1-σ)) * θ*(z1*h1)^(1-θ))^(-1/θ)* k1
        return eq
    end


    #log deviations
    T=ForwardDiff.gradient(loglineq1,[kss,kss,hss,zss,τhss,gss])
    a =[-kss*T[1]/(kss*T[1]),-kss*T[2]/(kss*T[1]),-hss*T[3]/(kss*T[1]),
    -zss*T[4]/(kss*T[1]),-τhss*T[5]/(kss*T[1]),-gss*T[6]/(kss*T[1])]
    #if ψ==0
    #    a[1],a[2:end]=-1,zeros(5)
    #end

    T=ForwardDiff.gradient(loglineq2,[kss,kss,kss,hss,hss,zss,τxss,gss,zss,τxss,gss])
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





    if nl == true
        function system!(eq,vector::Vector)
            #vector = rand(8)
            #eq= rand(8)
            B=vector[1:4]'
            D2 = vector[5:8]'

            eq[1:4] = a[2].*B .+ a[3].*D2 .+ [a[4] a[5] 0 a[6]]
            eq[5:8] = b[2].*B .+ b[3].*A.*B .+ b[3].*B*P .+ b[4].*D2 .+ b[5].*C[2].*B .+ b[5].*B*P.+
            [b[6] 0 b[7] b[8]].+[b[9] 0 b[10] b[11] ]*P
         return     eq
        end
        Sol = nlsolve(system!, ones(8),ftol = :1.0e-20, method = :trust_region , autoscale = true)
        D=ones(2,4)
        D[1,:]= Sol.zero[1:4]
        D[2,:]= Sol.zero[5:8]

    else
        #Implementing the code solve the system commented out above as a system of linear equations.
        TOP = hcat(a[2]*Matrix{Float64}(I,4,4),   a[3]*Matrix{Float64}(I,4,4))
        #Everything multiplying B concatenated with stuff multiplying D in the first equations
        BOTTOMLEFT =  b[2] .+ b[3].*A .+ b[3]*P .+ b[5].*C[2] .+ b[5]*P
        #Everything multiplying B in the last equations
        BOTTOM = hcat(BOTTOMLEFT',  b[4]*Matrix{Float64}(I,4,4)) #Concatenates with stuff multiplying D
        RHS = - vcat([a[4] a[5] 0 a[6]]',  ([b[6] 0 b[7] b[8]].+[b[9] 0 b[10] b[11] ]*P)') #Constant terms
        #Solving the system
        BD = (vcat(TOP,BOTTOM)\RHS)[:]
        D=ones(2,4)
        D[1,:]= BD[1:4]
        D[2,:]= BD[5:8]
    end



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



    ht([kss,zss,τhss,τxss,gss])


    function yt(vector::Vector)
        k,z,τh,τx,g = vector
        h = exp(ht(vector)+log(hss))
        y = k^θ * (z*h)^(1-θ)
        return y
    end

    T=ForwardDiff.gradient(yt,[kss,zss,τhss,τxss,gss])
    ycoefs = [kss*T[1]/yss,zss*T[2]/yss,τhss*T[3]/yss,τxss*T[4]/yss,gss*T[5]/yss]

    yt([kss,zss,τhss,τxss,gss])

    function xt(vector::Vector)
        k,z,τh,τx,g = vector
        k1 = exp(kt1(vector)+log(kss))
        x= (1+γn)*(1+γz)k1 - (1-δ)k

        return x
    end

    T=ForwardDiff.gradient(xt,[kss,zss,τhss,τxss,gss])
    xcoefs = [kss*T[1]/xss,zss*T[2]/xss,τhss*T[3]/xss,τxss*T[4]/xss,gss*T[5]/xss]

    C = [ycoefs[1] ycoefs[2] ycoefs[3] ycoefs[4] ycoefs[5];
    xcoefs[1] xcoefs[2] xcoefs[3] xcoefs[4] xcoefs[5];
    C[2,1] D[2,1] D[2,2] D[2,3] D[2,4];
    0 0 0 0 1]

return A,B,C
end
