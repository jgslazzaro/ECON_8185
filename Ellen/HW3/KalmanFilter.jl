using Statistics

function KalmanFilter(Y,A,B,C)
#Y: Observed variables


n = size(A)[1] #number of state variables
m = size(C)[1] #number of measurement variables
T = size(Y)[2] #Sample size

#Initializing the state variables
X = zeros(n,T)
#Variance initial guess
Σ = ones(n,n)*ones(n,n)'
d=10
while d>10^(-15)
    Σ1 = A*Σ*A' + B*B'
    d = maximum(abs.(Σ-Σ1))
    Σ=Σ1
end

#Run the Kalman filter algorithm (see Ljunqvist sargent)
a=ones(m,T)
Ω = C*Σ*C'
Σhist = copy(Σ)
for t = 1:T-1
    a[:,t] = Y[:,t] - C*X[:,t]
    K = A*Σ*C' / (C*Σ*C')
    X[:,t+1] = A*X[:,t] +K*a[:,t]
    Σ = B*B' + (A-K*C)*Σ*(A-K*C)'
    Σhist = [Σhist Σ]
    Ω = [Ω C*Σ*C']
end
a[:,T] = Y[:,T] - C*X[:,T]
Ω = [Ω C*Σ*C']

return X, a, Ω, Σhist
end

function KalmanSmoother(X,Σhist,A,B)
    #X are the filtered values from the Kalman filters
    #Ω contains all the Σ matrices from the Kalman filter
    T = size(X)[2] #Sample size
    m = size(Σhist)[1]

    Xsmooth = copy(X)
    Σsmooth = copy(Σhist)
    J=zeros(m,m*T)
    for t=T:-1:2
        J[:,m*(t-1)-(m-1):(m*(t-1))] = Σhist[:,m*(t-1)-(m-1):(m*(t-1))]*A'/(A*Σhist[:,m*(t-1)-(m-1):(m*(t-1))]*A'+B)
        Xsmooth[:,t-1] = X[:,t-1]+J[:,m*(t-1)-(m-1):(m*(t-1))]*(Xsmooth[:,t]- A*X[:,t])
        Σsmooth[:,m*(t-1)-(m-1):(m*(t-1))] =  Σhist[:,m*(t-1)-(m-1):(m*(t-1))] + J[:,m*(t-1)-(m-1):(m*(t-1))] * ( Σsmooth[:,m*t-(m-1):(m*t)]-A*Σhist[:,m*(t-1)-(m-1):(m*(t-1))]*A'+B)*J[:,m*(t-1)-(m-1):(m*(t-1))]'
    end

    return Xsmooth,Σsmooth,J
end

function EMSS(A,C,Xsmooth,Y,Σhist,Σsmooth,J)
    T = size(Xsmooth)[2]
    m = size(Σhist)[1]
    K = A*Σhist[:,m*(T)-(m-1):(m*(T))]*C' / (C*Σhist[:,m*(T)-(m-1):(m*(T))]*C')
    Σ12 = copy(Σhist)
    Σ12[:,m*(T)-(m-1):(m*(T))] = (I-K*C)*A*Σhist[:,m*(T-1)-(m-1):(m*(T-1))]
    S11 = copy(Σhist[:,m*(T)-(m-1):(m*(T))]).*0.0
    S10 = copy(S11)
    S00 = copy(S11)
    for t = T:-1:3
        Σ12[:,m*(t-1)-(m-1):(m*(t-1))] =  Σhist[:,m*(t-1)-(m-1):(m*(t-1))] *
        J[:,m*(t-2)-(m-1):(m*(t-2))]' +
        J[:,m*(t-1)-(m-1):(m*(t-1))]*(Σ12[:,m*(t)-(m-1):(m*(t))]-
        A*Σhist[:,m*(t-1)-(m-1):(m*(t-1))])*J[:,m*(t-2)-(m-1):(m*(t-2))]'
    end
    for t=2:T
        S11 = S11 .+ Xsmooth[t]*Xsmooth[t]'+ Σsmooth[:,m*(t)-(m-1):(m*(t))]
        S10 = S10 .+ Xsmooth[t]*Xsmooth[t-1]'+ Σ12[:,m*(t)-(m-1):(m*(t))]
        S00 = S00 .+ Xsmooth[t-1]*Xsmooth[t-1]'+ Σsmooth[:,m*(t-1)-(m-1):(m*(t-1))]

    end
    return S11,S10,S00,Σ12
end

function EM(Y,P,Q;steadystates= steadystates, params_calibrated = params_calibrated)
    A,B,C = State_Space(params_calibrated,steadystates, P,Q)
    for i = 1:10
        X, a, Ω, Σhist = KalmanFilter(Y,A,B,C)
        Xsmooth,Σsmooth,J = KalmanSmoother(X,Σhist,A,B)
        S11,S10,S00, Σ12 = EMSS(A,C,Xsmooth,Y,Σhist,Σsmooth,J)
        A = S10/S00
        B = (1/T) * (S11 - S10*(S00 \ S10'))
    end
    P = A[2:end,2:end]
    Q = B[2:end,2:end]
    return P,Q
end


function likelihood(Y,Ω,a;parallel = true)
    m,T = size(Y)
if parallel == false
    L = 0
for t=1:T
    L = L + (-0.5*m * log(2*π)-0.5*log(det(Ω[:,m*t-(m-1):(m*t)])) -
    0.5*a[:,t]'*(Ω[:,m*t-(m-1):(m*t)]\a[:,t]))
end
else
    L = zeros(T)
    Threads.@threads for t=1:T
        L[t] = (-0.5*m * log(2*π)-0.5*log(det(Ω[:,m*t-(m-1):(m*t)])) -
        0.5*a[:,t]'*(Ω[:,m*t-(m-1):(m*t)]\a[:,t]))
    end
    L = sum(L)
end
return L
end


function maxloglikelihood(vector::Vector;Y=Y)
    #ρg,ρx,ρh,ρz,σg,σx,σz,σh=vector
    ρg,ρx,ρh,ρz=vector
    #ρg,ρx,ρh,ρz,ρzg,ρzx,ρzh,ρhz,ρhx,ρhg,ρxz,ρxh,ρxg,ρgz,ρgx,ρgh,σg,σx,σz,σh,σzg,σzx,σzh,σhx,σhg,σxg,gss,τxss,τhss,zss = vector
    #ρg,ρx,ρh,ρz,ρzg,ρzx,ρzh,ρhz,ρhx,ρhg,ρxz,ρxh,ρxg,ρgz,ρgx,ρgh,σg,σx,σz,σh,σzg,σzx,σzh,σhx,σhg,σxg = vector

    #In matrix form
    P = [ρz ρzh ρzx ρzg;
    ρhz ρh ρhx ρhg ;
    ρxz ρxh ρx ρxg ;
    ρgz ρgh ρgx ρg]

    Q = [σz σzh σzx σzg;
    σzh σh σhx σhg ;
    σzx σhx σx σxg ;
    σzg σhg σxg σg]

    steadystates = gss,τxss,τhss,zss
    params_calibrated = [δ,θ,β,σ,ψ,γn,γz]


    A,B,C = State_Space(params_calibrated,steadystates, P,Q)
    X, a, Ω = KalmanFilter(Y,A,B,C)
    L = -likelihood(Y,Ω,a)
    return L
end
