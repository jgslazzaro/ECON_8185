using Distributions

function Tauchen(ρ,σ,Y,μ = 0,m = 3)
    #This function is to discretize an AR(1) process following Tauchen(1986) method
    # y_{t+1} = μ + ρy_t + ϵ
    #ϵ~N(0,σ^2)
    #Y is the number of y states
    if Y>1
        ybar = μ/(1-ρ)
        ymax= ybar + m*(σ^2/(1-ρ^2))^(1/2) #maximum y
        ymin= ybar - m*(σ^2/(1-ρ^2))^(1/2) #minimum y

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


function simMC(S,pdf,T,s0)
    #This function simulates a Markov chain
    #S possible states
    #pdF transition matrix of the states
    #n number of state variables controlled by pdf
    #T simulation length
    #s0 initial state
    nS::Int64 = length(S)
    ssim = fill(s0, T) #This is the states path
    r = rand(T)
    s::Int64=1
    #Simulating the economy
    for t=2:T
        s = findfirstn(S,ssim[t-1])
        ps = pdf[s,1]
        for i=1:nS
            if r[t]<=ps
                ssim[t]=S[i]
                break
            else
                ps+=pdf[s,i+1]
            end
        end
    end
    return ssim
end

function findfirstn(A,b)
    #this function finds the first index in which an element of A equals b
    s=0
    nA=length(A)
    for i=1:nA
        if A[i] == b
            s = i
            break
        end
    end
        return s
end

struct TransitionMatrix
    #this edfines the TransitionMatrix strcture
    #This function was adapted from Hori(2018) QuantEcon code:
    #https://notes.quantecon.org/submission/5bb58d1e11611400157fdc8d
    P::Matrix{Float64}       # 4x4
    Pz::Matrix{Float64}      # 2x2 aggregate shock
    Peps_gg::Matrix{Float64} # 2x2 idiosyncratic shock conditional on good to good
    Peps_bb::Matrix{Float64} # 2x2 idiosyncratic shock conditional on bad to bad
    Peps_gb::Matrix{Float64} # 2x2 idiosyncratic shock conditional on good to bad
    Peps_bg::Matrix{Float64} # 2x2 idiosyncratic shock conditional on bad to good
end

function create_transition_matrix(;ug::AbstractFloat=0.04, #Unemployment rate in good times
    ub::AbstractFloat=0.1, #Unemployment rate in bad times
    zg_ave_dur::Real=8, #average duration of good period
    zb_ave_dur::Real=8, #average duration of bad period
    ug_ave_dur::Real=1.5, #average duration of unemployment period in good times
    ub_ave_dur::Real=2.5, #average duration of unemployment period in bad times
    puu_rel_gb2bb::Real=1.25, #imposed conditions
    puu_rel_bg2gg::Real=0.75)  #imposed conditions
    #Compute the transition matrices
    #This function was adapted from Hori(2018) QuantEcon code:
    #https://notes.quantecon.org/submission/5bb58d1e11611400157fdc8d
    # probability of remaining in good state
    pgg = 1-1/zg_ave_dur
    # probability of remaining in bad state
    pbb = 1-1/zb_ave_dur
    # probability of changing from g to b
    pgb = 1-pgg
    # probability of changing from b to g
    pbg = 1-pbb

    # prob. of 0 to 0 cond. on g to g
    p00_gg = 1-1/ug_ave_dur
    # prob. of 0 to 0 cond. on b to b
    p00_bb = 1-1/ub_ave_dur
    # prob. of 0 to 1 cond. on g to g
    p01_gg = 1-p00_gg
    # prob. of 0 to 1 cond. on b to b
    p01_bb = 1-p00_bb

    # prob. of 0 to 0 cond. on g to b
    p00_gb=puu_rel_gb2bb*p00_bb
    # prob. of 0 to 0 cond. on b to g
    p00_bg=puu_rel_bg2gg*p00_gg
    # prob. of 0 to 1 cond. on g to b
    p01_gb=1-p00_gb
    # prob. of 0 to 1 cond. on b to g
    p01_bg=1-p00_bg

    # prob. of 1 to 0 cond. on  g to g
    p10_gg=(ug - ug*p00_gg)/(1-ug)
    # prob. of 1 to 0 cond. on b to b
    p10_bb=(ub - ub*p00_bb)/(1-ub)
    # prob. of 1 to 0 cond. on g to b
    p10_gb=(ub - ug*p00_gb)/(1-ug)
    # prob. of 1 to 0 cond on b to g
    p10_bg=(ug - ub*p00_bg)/(1-ub)
    # prob. of 1 to 1 cond. on  g to g
    p11_gg= 1-p10_gg
    # prob. of 1 to 1 cond. on b to b
    p11_bb= 1-p10_bb
    # prob. of 1 to 1 cond. on g to b
    p11_gb= 1-p10_gb
    # prob. of 1 to 1 cond on b to g
    p11_bg= 1-p10_bg

                #b0         b1        g0           g1
    P = [pbb*p00_bb pbb*p01_bb pbg*p00_bg pbg*p01_bg;
         pbb*p10_bb pbb*p11_bb pbg*p10_bg pbg*p11_bg;
         pgb*p00_gb pgb*p01_gb pgg*p00_gg pgg*p01_gg;
         pgb*p10_gb pgb*p11_gb pgg*p10_gg pgg*p11_gg]

    Pz=[pbb pbg;
        pgb pgg]
    Peps_gg=[p00_gg p01_gg;
            p10_gg p11_gg]
    Peps_bb=[p00_bb p01_bb;
            p10_bb p11_bb]
    Peps_gb=[p00_gb p01_gb;
            p10_gb p11_gb]
    Peps_bg=[p00_bg p01_bg;
            p10_bg p11_bg]

    transmat=TransitionMatrix(P, Pz, Peps_gg, Peps_bb, Peps_gb, Peps_bg)
    return transmat
end

function idioshocks(zsim::Array{Float64,1},transmat::TransitionMatrix;N=N::Int64,
    Z=Z::Array{Float64,1},E=E::Array{Float64,1}, e0 = E[1]::Float64,ug=ug::Float64,ub=ub::Float64)
    #This function finds idiosincratic shocks consistent with aggregate shocks distribution
    T::Int64 = length(zsim)
    nE::Int64 = length(E)
    nZ::Int64 =length(E)
    esim::Array{Float64,2} = fill(0.0,N,T) #This is the states path
    pdf::Array{Float64,1} = ones(2)
    ps::Float64=0.0
    u::Int64=0
    r::Array{Float64,2} = rand(N,T) #Random draws

    #First period:
    for n=1:N
        if zsim[1]==Z[1]
            if r[n,1]>ub
                esim[n,1] = E[2]
            else
                esim[n,1] = E[1]
            end
        else
            if r[n,1]>ug
                esim[n,1] = E[2]
            else
                esim[n,1] = E[1]
            end
        end
    end

    #Simulating the economy
    for n = 1:N
      for t=2:T
            e0 = findfirstn(E,esim[t-1])
            z0 = findfirstn(Z,esim[t-1])
            z1 = findfirstn(Z,esim[t])
            if z0==Z[1] && z1==Z[1] #conditional on bad to bad and e0
                pdf = transmat.Peps_bb[e0,:]
            elseif z0==Z[1] && z1==Z[2] #conditional on bad to good and e0
                pdf = transmat.Peps_bg[e0,:]
            elseif z0==Z[2] && z1==Z[1]#conditional on good to bad and e0
                pdf = transmat.Peps_gb[e0,:]
            else #conditional on good to good and e0
                pdf = transmat.Peps_gg[e0,:]
            end
            ps =  pdf[1]
            for i=1:nE
                if r[n,t]<=ps
                    esim[n,t]=E[i]
                    break
                else
                    ps+=pdf[i+1]
                end
            end
        end
    end
    return esim
end
