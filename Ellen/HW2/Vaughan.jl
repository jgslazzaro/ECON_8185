using LinearAlgebra
#This function implements the Vaughan's method

function Vaughan_dist(Atilde,Ahat,B,By,Q,R,W)
    L=size(A)[1]
    ℋ = [inv(Ahat)  inv(Ahat)*B*inv(R)*By';Q*inv(Ahat) Q*inv(Ahat)*B*inv(R)*By'+Atilde'] #This is the coefficient matrix.
    V= eigen(ℋ).vectors #Take the eigenvector matrix The first
    #Note that Julia puts the eigenvalues out of the unit circle in the bottom of the matrix,
    #while in the lecture notes they are at the top
    P = V[L+1:end,L+1:end] * inv(V[1:L,L+1:end])  #Get the P matrix
    F=inv(R+B'*P*B) * B' * P * A +inv(R)*W' #Finally, compute F
    return P, F
end
