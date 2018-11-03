using LinearAlgebra
function Vaughan(A,B,Q,R,W,β,γ1=10^(-5),γ2=10^(-5),p=1)



#Guess initial value for P
P=ones(size(A)[2],size(A)[1])
F=ones(size(R)[1],size(W)[1])

#initialize loop:
distP = 10; distF = 10; i=1

ℋ1 = [A zeros(size(A)); Q]


Λ = eigen(ℋ).values
V =eigen(ℋ).vectors
Wv= inv(V)


while distP>=γ1.*opnorm(P,p) || distF>=γ2*opnorm(F,p)
    #global P, F, i, distP, distF, P1, F1

    P1 = Q + A'*P*A - A'*P*B*inv(R+B'* P *B)*B'*P*A
    F1 = inv(R+B'*P*B) * B' * P * A
    distP = opnorm(P1-P,p)
    distF = opnorm(F1-F,p)
    i=i+1

    P=copy(P1)
    F = copy(F1)
end

F = F+inv(R)*W'

return P, F
end
