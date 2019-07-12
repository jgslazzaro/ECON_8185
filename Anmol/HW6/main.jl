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


nμ = 50
nB = 120
B = range(-1.0,stop=1.5,length = nB)
μ = range(0,stop = .55,length = nμ)

nΠ = length(Π)
nG = length(G)


Wgrid = ones(nB,nμ,nG)

consumption(n,g) = n - g

function V(n,γ,b,g,μ)
    c = consumption(n,g)
    UC = Uc(c,n)
    UN = Un(c,n)
    V = U(c,n) +μ*b*UC+γ*(UC*(c-b)+n*UN)
    return V
end

#Γ = range(0,stop = 5,length = nΓ)

nstar(γ,b,g,μ) = optimize(n-> -V(n,γ,b,g,μ) ,g,1.0).minimizer



Vstar(γ,b,g,μ) = V(nstar(γ,b,g,μ),γ,b,g,μ)

Vstar(0.1,0.1,0.1,.1)
nstar(0.1,0.1,0.1,.1)
#Guess for value function:

itpW = LinearInterpolation((B,μ,G),Wgrid,extrapolation_bc=Line())
W(b,μ,g) = itpW(b,μ,g)


function EW(b1,μ1,g;Π=Π)
    g0 = findfirst(G.==g)
    expected::Float64 =0.0
    for gi = 1:nG
        expected += Π[g0,gi]*W(b1,μ1,G[gi])
    end
    return expected
end

function EW(b1,μ1,g,Wgrid;Π=Π)
    g0 = findfirst(G.==g)
    bi = findfirst(B.==b1)
    μi = findfirst(μ.==μ1)

    expected::Float64 =0.0
    for gi = 1:nG
        expected += Π[g0,gi]*Wgrid[bi,μi,gi]
    end
    return expected
end

EW(B[1],μ[1],G[1],Wgrid)



function SPFE(Wgrid::Array{Float64,3},B,G,μ;nB=nB,nG=nG,nμ=nμ,β=β )
    Wgrid1::Array{Float64,3}=copy(Wgrid)
    objective::Array{Float64,2} = ones(nμ,nB)
    policy::Array{Int64,4} = ones(Int64,nB,nμ,nG,2)
    dist::Float64 =1.0
    Vstargrid::Array{Float64,4} = ones(nB,nμ,nG,nμ)
    EWgrid::Array{Float64,3} = ones(nB,nμ,nG)

        for γi=1:nμ,gi = 1:nG, μi = 1:nμ , bi = 1:nB
            Vstargrid[bi,μi,gi,γi] = Vstar(μ[γi],B[bi],G[gi],μ[μi])
        end


    while dist>1e-5
        #global Wgrid1,Wgrid,policy,dist,W
        for gi = 1:nG, γi = 1:nμ,b1 = 1:nB
            EWgrid[b1,γi,gi] = EW(B[b1],μ[γi],G[gi],Wgrid)
        end
            for gi = 1:nG,μi = 1:nμ ,bi = 1:nB
                for γi=1:nμ, b1i=1:nB
                    objective[γi,b1i] = Vstargrid[bi,μi,gi,γi] +β * EWgrid[b1i,γi,gi]
                end
                ob2,indsB =findmax(objective,dims=2)
                Wgrid1[bi,μi,gi],indsμ = findmin(ob2)
                policy[bi,μi,gi,2] = indsμ[1]
                policy[bi,μi,gi,1] = indsB[indsμ[1]][2]
            end
        dist = maximum(abs.(Wgrid.-Wgrid1))
        println("distance is $(dist)")
        Wgrid = 0.75*Wgrid1+0.25*Wgrid
    end
    return Wgrid,policy
end

Wgrid,policy = SPFE(Wgrid,B,G,μ;nB=nB,nG=nG,nμ=nμ)

maximum(policy[:,:,:,1])
maximum(policy[:,:,:,2])


T = 20
Ghist = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
b0 = .50
μ0 = 0.0

Bhist = fill(findfirst(B.>=b0),T)
μhist = fill(findfirst(μ.==μ0),T)
nhist = ones(T)


for t = 2:T
    Bhist[t] = policy[Bhist[t-1],μhist[t-1],Ghist[t-1],1]
    μhist[t] = policy[Bhist[t-1],μhist[t-1],Ghist[t-1],2]
end

for t=1:T
    nhist[t] = nstar(μ[μhist[t]],B[Bhist[t]],G[Ghist[t]],μ[μhist[t]])
end

Bhist = B[Bhist]
Ghist = G[Ghist]
μhist = μ[μhist]

chist = nhist .-Ghist

tauhist = 1 .- Un.(chist,nhist) ./ Uc.(chist,nhist)

using Plots


titles = hcat("Government Debt","Government Spending","Tax Rate","Consumption","labor", "Lagrange Multiplier")
p = plot(size = (920, 750), layout = grid(3, 2),
         xaxis=(0:T), grid=false, titlefont=Plots.font("sans-serif", 10))
plot!(p, title = titles, legend=false)
plot!(p[1], Bhist, marker=:utriangle, color=:blue)
plot!(p[2], Ghist, marker=:utriangle, color=:blue)
plot!(p[3], tauhist, marker=:utriangle, color=:blue)
plot!(p[4], chist, marker=:utriangle, color=:blue)
plot!(p[5], nhist, marker=:utriangle, color=:blue)
plot!(p[6], μhist, marker=:utriangle, color=:blue)
