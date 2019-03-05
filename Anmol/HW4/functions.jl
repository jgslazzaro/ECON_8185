using Distributions, Optim, ApproxFun, NLsolve

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


function utility(c,l,η,μ)
    if (c<=0) || (η<1 && l==0)
        u=-Inf
    elseif μ==1
        u = log(c^η * l^(1-η))
    else
        u = ((c^η * l^(1-η))^(1-μ) )/ (1-μ)
    end
    return u
end


function VFI(A,E,Z,Kbar,Lbar,r,w,τy,T,β,η,μ)

    nZ= length(Z)
    nA = length(A)
    nE = length(E)
    u = ones(nA,nE,nZ,nA)


    #Solving the labor problem for each level of asset today and tommorow
    #I tested and it is faster than including this in the main while loop
    c(l,a,e,a1,z) = (1-τy)*E[e]*w(z,Kbar,Lbar)*(1-l)+T+(1+(1-τy)*r(z,Kbar,Lbar))*A[a]-A[a1]
    if η<1
        policy_l1=ones(nA,nE,nZ,nA)
        for a=1:nA,e=1:nE,a1=1:nA,z=1:nZ
                maxu(l) = -utility(c(l,a,e,a1,Z[z]),l,η,μ)
                maxU = optimize(maxu, 0.0, 1.0)
                u[a,e,z,a1] = -maxU.minimum
                policy_l1[a,e,z,a1] = maxU.minimizer
        end
    else
        for a=1:nA,e=1:nE,a1=1:nA,z=1:nZ
            if E[e]>0
                u[a,e,z,a1] = utility(c(0,a,e,a1,z),0,η,μ)
            else
                u[a,e,z,a1] = utility(c(1,a,e,a1,z),1,η,μ)
            end
        end
    end

    #VFI
    #Initial value function guess:
        V  = ones(nA,nE,nZ)
        #preallocation
        policy_a = Array{Int64,3}(undef,nA,nE,nZ)
        #loop stuff
        distance = 1
        tol =10^(-7)
        #iterating on Value functions:
        while distance >= tol
            Vf = copy(V) #save V to compare later
            EVf = copy(V)
            #find the expected Vf: E[V(k,z')|z] = π(z1|z)*V(k,z1) +π(z2|z)*V(k,z2)
            for z=1:nZ
                EVf[:,:,z]= (pdfZ*pdfE[:,:,z]*Vf[:,:,z]')'  #The transposing fix the dimensions so it works!
            end
            #maximization loop, for each state I find the a1 that maximizes the value function
            #This is discretized, no interpolation. It could be faster and smarter,
            #I'll get back to this
            for a in 1:nA, e in 1:nE, z in 1:nZ
                V[a,e,z] , policy_a[a,e,z] = findmax(u[a,e,z,:] .+ β * EVf[:,e,z]) #Bellman Equation
            #The if below is necessary because in the worst states, consumption is negative,
            #no matter what choice of assets. So, in these casesthe findmax function above would
            #choose weird values, so I set the policy functions at the initial values at the grid
                if u[a,e,z,policy_a[a,e,z]] == -Inf
                    #policy_a[a,e] = 1
                end
            end
         distance = maximum(abs.(V-Vf))
        end
        #Finally, find labor policy:
        policy_l = zeros(nA,nE,nZ)
        if η<1
            for a in 1:nA, e in 1:nE, z=1:nZ
                policy_l[a,e,z] = policy_l1[a,e,z,policy_a[a,e,z]]
                #it is the policy for labor we found before, with the addition of the
                #policy function for assets
            end
        else
            for e = 1:nE,z =1:nZ
                if E[e]<=0
                    policy_l[:,e,z] = ones(nA)
                end
            end
        end

        #and for consumption
            policy_c=ones(nA,nE,nZ)
        for a=1:nA,e=1:nE, z=1:nZ
            policy_c[a,e,z] = (1-τy)*E[e]*w(z,Kbar,Lbar)*(1-policy_l[a,e,z])+T+(1+(1-τy)*r(z,Kbar,Lbar))*A[a]-A[policy_a[a,e,z]]
        end
return policy_a, policy_c, policy_l
end


function KrusselSmith(A,E,Z,β,η,μ)

    nZ = length(Z)
    nA = length(A)
    nE = length(E)
    #Following Violante slides steps:
    #Step 1: Specify a functional form for the law of motion, for example, linear:

    K1(K,Bk) = Bk[1] + Bk[2]*K
    L1(K,Bl) = Bl[1] + Bl[2]*K

    #Step 2: Guess the matrices of coefﬁcients
    Bk = [0 1;0 1] #Initial guess for the coefficients
    Bl = [0 1;0 1]

    N = ones(nZ)

    #Step 3: Specify how prices depend on K
    r(z,K,L) = θ*z * K^(θ-1) * L^θ + 1 - δ
    w(z,K,L) = (1-θ)*z * K^θ * L^(-θ)

    #Step 4: Solve the household problem and obtain the decision rule with standard methods.
    #Since I'm using only one moment, I won't create a grid for m.

    policy_a, policy_c, policy_l = VFI(A,E,Z,1,1, r, w,0,0,β,η,μ)



end
