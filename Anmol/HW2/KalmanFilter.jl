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
for t = 1:T-1
    a[:,t] = Y[:,t] - C*X[:,t]
    K = A*Σ*C' / (C*Σ*C')
    X[:,t+1] = A*X[:,t] +K*a[:,t]
    Σ = B*B' + (A-K*C)*Σ*(A-K*C)'
    Ω = [Ω C*Σ*C']
end
a[:,T] = Y[:,T] - C*X[:,T]
Ω = [Ω C*Σ*C']

return X, a, Ω
end

function likelihood(Y,Ω,a)
    m,T = size(Y)
    L = 0
for t=1:T
    iteration = t
    L = L + (-T*m/2 * log(2*π)-0.5*log(det(Ω[:,m*t-(m-1):(m*t)])) -
    0.5*a[:,t]'*(Ω[:,m*t-(m-1):(m*t)]\a[:,t]))
end

return L
end


function maxloglikelihood(vector::Vector)
    ρz,ρh,ρx,ρg,σz,σh,σx,σg = vector

    #In matrix form
    P = [ρz 0 0 0;
    0 ρh 0 0 ;
    0 0 ρx 0 ;
    0 0 0 ρg]
    Q = [σz 0 0 0;
    0 σh 0 0 ;
    0 0 σx 0 ;
    0 0 0 σg]
    params_calibrated = [δ,θ,β,σ,ψ,γn,γz,gss,τxss,τhss,zss]
    A,B,C = State_Space(params_calibrated, P,Q)
    X, a, Ω = KalmanFilter(Y,A,B,C)
    L = -likelihood(Y,Ω,a)
    return L
end

function maxloglikelihood2(vector::Vector)
    ρz,σz = vector
    #Only Productivity shocks
    #In matrix form
    P = [ρz 0 0 0;
    0 0 0 0 ;
    0 0 0 0 ;
    0 0 0 0]
    Q = [σz 0 0 0;
    0 0 0 0 ;
    0 0 0 0 ;
    0 0 0 0]
    params_calibrated = [δ,θ,β,σ,ψ,γn,γz,gss,τxss,τhss,zss]
    A,B,C = State_Space(params_calibrated, P,Q)
    A = A[1:2,1:2]
    B = B[1:2,1:2]
    C = C[1:3,1:2]
    U = Y[1:3,:]
    X, a, Ω = KalmanFilter(U,A,B,C)
    L = -likelihood(U,Ω,a)
    return L
end
