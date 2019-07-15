using Optim, Interpolations

#Defining parameters
const β = 0.9
const ψ = 0.69
const Π = 0.5 * ones(2, 2)
const G = [0.1, 0.2]
const Θ = ones(2)

# Derivatives of utility function
U(c,n;ψ=ψ) = log(c) + ψ * log(1 - n)
Uc(c,n) = 1 ./ c
Ucc(c,n) = -c.^(-2.0)
Un(c,n;ψ=ψ) = -ψ ./ (1.0 .- n)
Unn(c,n;ψ=ψ) = -ψ ./ (1.0 .- n).^2.0


nB = 10
B= range(0.0,stop = 2.0, length = nB)
nG = length(G)

nμ = 10
μ = range(0,stop = 1.0,length = nμ)


consumption(n,g) = n - g

function V(n,γ,b,g,μ)
    c = consumption(n,g)
    UC = Uc(c,n)
    UN = Un(c,n)
    V = U(c,n) +μ*b*UC+γ*(UC*(c-b)+n*UN)
    return V
end


nstar(γ,b,g,μ) = optimize(n-> -V(n,γ,b,g,μ) ,g,1.0).minimizer



Vstar(γ,b,g,μ) = V(nstar(γ,b,g,μ),γ,b,g,μ)

Vstar(0.1,0.1,0.1,.1)
nstar(0.1,0.1,0.1,.1)
#Guess for value function:

Wgrid = ones(nB,nμ,nG)
itpW = LinearInterpolation((B,μ,G),Wgrid,extrapolation_bc=Line())
W(b,μ,g) = itpW(b,μ,g)


function EW(b1,μ1,g;Π=Π,W=W)
    g0 = findfirst(G.==g)
    expected::Float64 =0.0

    for gi = 1:nG
        expected += Π[g0,gi]*W(b1,μ1,G[gi])
    end
    return expected
end



function findSP(b,g,μ0,Wint;B=B,μ=μ)
        maxfunc(γ1) =  -optimize(b1->-(Vstar(γ1,b,g,μ0) +β * EW(b1,γ1,g,W=Wint)),B[1],B[end]).minimum
        argmaxfunc(γ1) = optimize(b1->-(Vstar(γ1,b,g,μ0) +β * EW(b1,γ1,g,W=Wint)),B[1],B[end]).minimizer
        minfunc = optimize(maxfunc,μ[1],μ[end])
        γ1 = minfunc.minimizer
        b1 = argmaxfunc(γ1)
        return minfunc.minimum, γ1, b1
end




function SPFE(Wgrid::Array{Float64,3},B,G,μ;nB=nB,nG=nG,nμ=nμ,β=β )
    Wgrid1::Array{Float64,3}=copy(Wgrid)
    objective::Array{Float64,2} = ones(nμ,nB)
    policy::Array{Float64,4} = ones(nB,nμ,nG,2)
    dist::Float64 =1.0
    itpW = LinearInterpolation((B,μ,G),Wgrid,extrapolation_bc=Line())
    Wint(b,μ,g) = itpW(b,μ,g)

    while dist>1e-5
        #global Wgrid1,Wgrid,policy,dist,W
            for gi = 1:nG,μi = 1:nμ,bi = 1:nB
                Wgrid1[bi,μi,gi],policy[bi,μi,gi,2],policy[bi,μi,gi,1] = findSP(B[bi],G[gi],μ[μi],Wint)
            end

        dist = maximum(abs.(Wgrid.-Wgrid1))

        println("distance is $(dist)")
        Wgrid =0.75*Wgrid1 + 0.25*Wgrid
        itpW = LinearInterpolation((B,μ,G),Wgrid,extrapolation_bc=Line())
    end
    return Wgrid,policy
end


nB = 15
B= range(-.5,stop =1.2, length = nB)
nG = length(G)

nμ = 15
μ = range(0,stop = 1.5,length = nμ)
Wgrid = ones(nB,nμ,nG)


Wgrid,policy = SPFE(Wgrid,B,G,μ)


policyB = LinearInterpolation((B,μ,G),policy[:,:,:,1])#,extrapolation_bc=Line())
policyμ = LinearInterpolation((B,μ,G),policy[:,:,:,2])#,extrapolation_bc=Line())

T = 20
Ghist = G[[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]]
b0 = .50
μ0 = 0.0

Bhist = fill(b0,T)
μhist = fill(μ0,T)
for t = 2:T
    Bhist[t] = policyB(Bhist[t-1],μhist[t-1],Ghist[t-1])
    μhist[t] = policyμ(Bhist[t-1],μhist[t-1],Ghist[t-1])
end

t=2


using Plots

plot(Bhist)
plot!(μhist)
