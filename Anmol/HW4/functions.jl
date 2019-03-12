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

function uc(c,l,η,μ)
    #derivative of u with respect to c
    uc = (η * c^(η-1) * l^(1-η)) * (c^η * l^(1-η))^(-μ)
    return ul
end

function ul(c,l,η,μ)
    #derivative of u with respect to l
    ul = ((1-η) * c^η * l^(-η))*(c^η * l^(1-η))^(-μ)
    return ul
end

function policies(A,E,Z,K,L)

return policy_a, policy_c, policy_l
end

function KrusselSmith(A,E,Z,K,L,pdfE,pdfZ;β=β,η=η,μ=μ)

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
    #

    policy_a, policy_c, policy_l = policies(A,E,Z,K,L)



end
