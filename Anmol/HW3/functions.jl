using Distributions, Optim, ApproxFun
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
        u=-10^10
    elseif μ==1
        u = log(c^η * l^(1-η))
    else
        u = ((c^η * l^(1-η))^(1-μ) )/ (1-μ)
    end
    return u
end


function VFI(A,E,r,w)

    nA = length(A)
    nE = length(E)
    u = ones(nA,nE,nA)
    policy_l1=ones(nA,nE,nA)
    #Solving the labor problem

    c(l,a,e,a1) = (1-τy)*E[e]*w*(1-l)+T+(1+(1-τy)*r)*A[a]-A[a1]
    for a=1:nA,e=1:nE,a1=1:nA
            maxu(l) = -utility(c(l,a,e,a1),l,η,μ)
            maxU = optimize(maxu, 0.0, 1.0)
            u[a,e,a1] = -maxU.minimum
            policy_l1[a,e,a1] = maxU.minimizer
    end

    #VFI
    #Initial value function guess:
        V  = zeros(nA,nE)
        #preallocation
        policy_a = Array{Int64,2}(undef,nA,nE)
        #loop stuff
        distance = 1
        tol =10^(-10)
        #iterating on Value functions:
        while distance >= tol
            #global distance, tol, V, policy_a
            Vf = copy(V) #save V to compare later
            #find the expected Vf: E[V(k,z')|z] = π(z1|z)*V(k,z1) +π(z2|z)*V(k,z2)
            EVf= (pdfE*Vf')'  #The transposing fix the dimensions so it works!
            for a in 1:nA, e in 1:nE
                V[a,e] , policy_a[a,e] = findmax(u[a,e,:] .+ β * EVf[:,e]) #Bellman Equation
            end
         distance = maximum(abs.(V-Vf))
        end
        #Finally, find labor policy:

        policy_l = Array{Float64,2}(undef,nA,nE)
        for a in 1:nA, e in 1:nE
            policy_l[a,e] = policy_l1[a,e,policy_a[a,e]]
        end

        #and for consumption
            policy_c=ones(nA,nE)
        for a=1:nA,e=1:nE

            policy_c[a,e] = (1-τy)*E[e]*w*(1-policy_l[a,e])+T+(1+(1-τy)*r)*A[a]-A[policy_a[a,e]]

        end
return policy_a, policy_c, policy_l
end


function ayiagary(A,E,r0,w0) #Given an initial interest rate and initial grids,
    #returns the distribution of states and policy functions, and equilibirum interest rate.
    r = r0
    K = ((r+δ)/θ)^(1/(θ-1))
    w =  (1-θ)*K^θ
    nA = length(A)
    nE = length(E)
    dist_r = 1.0
    dist_k = 1.0
    λ = ones(nA,nE).*1/(nA*nE)
    policy_a, policy_c, policy_l = VFI(A,E,r,w)
    while maximum([dist_r,dist_k])>10^-8
            #global dist_r, r, dist_k

            dist = 10
            iterations = 0

            while dist>10^-10
                    #global λ,dist,iterations
                    λ1=zeros(nA,nE)
                    for a1=1:nA,e1=1:nE
                            for a=1:nA
                                    for e =1:nE
                                        if a1 .==policy_a[a,e]
                                                λ1[a1,e1] +=  λ[a,e].*pdfE[e,e1]
                                        end
                                    end
                            end
                    end
                    dist = maximum(abs.(λ1-λ))
                    λ = λ1
                    iterations +=1
                    #println(iterations)
            end
            K = ((r+δ)/θ)^(1/(θ-1))
            k= sum(λ .* A[policy_a])
            if k>0
                r1=1/2 * (r + (θ*k^(θ-1)-δ) )
            else
                r1 = 1/2 *r
            end
            dist_r = abs(r1-r)
            dist_k = abs(K-k)

            r=r1
            w =  (1-θ)*K^θ
            policy_a, policy_c, policy_l = VFI(A,E,r,w)
    end

return  λ,r,w, policy_a, policy_c, policy_l
end
function ayiagary_supply(A,E,r0,w0,R) #Given an initial interest rate and initial grids,
    #returns the distribution of states and policy functions, and equilibirum interest rate.
    r = r0
    K = ((r+δ)/θ)^(1/(θ-1))
    w =  (1-θ)*K^θ
    nA = length(A)
    nE = length(E)
    dist_r = 1.0
    dist_k = 1.0
    λ = ones(nA,nE).*1/(nA*nE)
    policy_a, policy_c, policy_l = VFI(A,E,r,w)
    #while maximum([dist_r,dist_k])>10^-8
    k=zeros(R)
    policy_a, policy_c, policy_l = VFI(A,E,0,w)
    i = 1
    for r = range(0.0,stop = 1.0,length = R)
            #global dist_r, r, dist_k
            policy_a, policy_c, policy_l = VFI(A,E,r,w)
            dist = 10
            iterations = 0

            while dist>10^-10
                    #global λ,dist,iterations
                    λ1=zeros(nA,nE)
                    for a1=1:nA,e1=1:nE
                            for a=1:nA
                                    for e =1:nE
                                        if a1 .==policy_a[a,e]
                                                λ1[a1,e1] +=  λ[a,e].*pdfE[e,e1]
                                        end
                                    end
                            end
                    end
                    dist = maximum(abs.(λ1-λ))
                    λ = λ1
                    iterations +=1
                    #println(iterations)
            end
            #K = ((r+δ)/θ)^(1/(θ-1))
            k[i]= sum(λ .* A[policy_a])
            i=i+1
            #=if k>0
                r1=1/2 * (r + (θ*k^(θ-1)-δ) )
            else
                r1 = 1/2 *r
            end
            dist_r = abs(r1-r)
            dist_k = abs(K-k)

            r=r1 =#
            w =  (1-θ)*K^θ
            #policy_a, policy_c, policy_l = VFI(A,E,r,w)
    end

return  k
end
