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
nS = length(S)
ssim = fill(s0, T)
r = rand(T)
s=1
#Simulating the economy
for t=2:T
    s = findfirstn(S,ssim[t-1])
    ps = pdf[s,1]
    for i=1:nS
        if r[t]<=ps
            ssim[t]=S[i][:]
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
