include("functions.jl")
#include("VFI_KS.jl")
#include("Euler_KS.jl")
include("Carrol_KS.jl")


policy = zeros(nA,nE,nK,nH,nZ,2)
for a=1:nA ,e=1:nE,k=1:nK,h=1:nH,z=1:nZ
    policy[a,e,k,h,z,:] = [0.9*A[a],E[e] * lbar-0.1]
end
    #Guess for policy functions
itpn = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,2],
extrapolation_bc=Line())
itpa = LinearInterpolation((A,E,K,H,Z),policy[:,:,:,:,:,1],
extrapolation_bc=Line())
policy_a(a,e,k,h,z) = itpa(a,e,k,h,z)
policy_n(a,e,k,h,z;μ=μ) = (e>0)*((μ==1.0)*nstar(a,policy_a(a,e,k,h,z),k,h,z) + (μ!=1.0)*itpn(a,e,k,h,z))
policygrid = ENDOGENOUSGRID_KS(A,E,Z,transmat,states,K, H,b,d;policy= policy,update_policy=update_policy,tol = tol)
