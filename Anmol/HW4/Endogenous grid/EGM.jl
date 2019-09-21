include("CRRA_utility.jl")
include("functions.jl")
using  Interpolations,NLsolve
η = 1.0
β = 0.95
μ = 1.0
ρ = 0.8
σ = 0.1

#R=1/β
#defining grids for A and y
nA = 25
A = range(-5,stop =1, length =nA)

nY = 20
pdfY, y = Tauchen(ρ,σ,nY)
y = exp.(y)


R = 1/β-0.025

#No labor choice
function EGM(A::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
    y::Array{Float64,1},pdfY::Array{Float64,2};R::Float64=R,damp::Float64=0.5)

    #Guess for consumption
    cgrid1 =zeros(nA,nY)
    for a1 =1:nA, y0=1:nY
        cgrid1[a1,y0] = R*A[a1]+y[y0]
    end
    #initializing consumption implied by assets tommorow
    ctil = ones(nA,nY)
    B = zeros(nA,nY)
    astar = zeros(nA,nY)
    distance = 1.0

    while distance >1e-6
        cgrid = copy(cgrid1)
        for y0=1:nY
            for a1 =1:nA
                B[a1,y0] = β*   sum(pdfY[y0,:] .* R .* uc.(cgrid[a1,:],.0))
                ctil[a1,y0] = B[a1,y0].^(-1/μ)
                astar[a1,y0] = (ctil[a1,y0]+A[a1] -y[y0])/R
            end

            itpc0 = LinearInterpolation(astar[:,y0],ctil[:,y0], extrapolation_bc=Line())
            for a1 =1:nA
                if A[a1] >= astar[1,y0]
                    cgrid1[a1,y0] = itpc0(A[a1])
                else
                    cgrid1[a1,y0] = R*A[a1]-A[1] +y[y0]
                end
            end
        end
        distance = maximum(abs.(cgrid1.-cgrid))
        cgrid1 = damp*cgrid1 .+ (1-damp)*cgrid

        println("distance $(distance)")
    end
    itp = LinearInterpolation((A,y),cgrid1, extrapolation_bc=Line())
    c(a,y) = itp(a,y)
    return c,cgrid1
end

c,cgrid1 = EGM(A,y,pdfY;R=R,damp=0.5)

a1(a,y; R = R) = R*a - c(a,y) +y

using Plots
plot(A,[c.(A,y[2]), c.(A,y[end-1])],label=["Low y", "High y"],title = "Consumption policy")
savefig("EGM_cons.jl")
plot(A,[A, a1.(A,y[2]), a1.(A,y[end-1])],label=["45","low y", "high y"],legend=:bottomright,title = "Asset Policy")
savefig("EGM_capital.jl")

#= weird crap down here
w = 1.0

function euler!(eq::Array{Float64,1},x,y0::Int64,a1::Int64,cgrid::Matrix{Float64})
    c0,n = x
    B = β* sum(pdfY[y0,:] .* R .* uc.(cgrid[a1,:],1.0.-ngrid1[a1,:]))

    eq[1] = uc.(c0,1.0-n)-  B
    eq[2] = ul.(c0,1.0-n)- w*y0*uc.(c0,1.0-n)
    return eq
end

eq = euler!([1.0,1.0],[1.0,0.7],5,3,cgrid)


w=1
#with labor choice
#function EGM(A::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
#    y::Array{Float64,1},pdfY::Array{Float64,2};R::Float64=R,damp::Float64=0.5)

#Guess for labor
ngrid1 =ones(nA,nY)
#Guess for consumption
cgrid1 =zeros(nA,nY)
for a1 =1:nA, y0=1:nY
    cgrid1[a1,y0] = R*A[a1]+y[y0]*w*ngrid1[a1,y0]
end

#initializing consumption implied by assets tommorow
ctil = ones(nA,nY)
ntil = ones(nA,nY)
B = zeros(nA,nY)
astar = zeros(nA,nY)
distance = 1.0

while distance >1e-6
global ctil,cgrid1,B,Astar,distance,ngrid1
cgrid = copy(cgrid1)
ngrid = copy(ngrid1)

for y0=1:nY
for a1 =1:nA
global ctil,cgrid1,B,Astar,distance,ngrid1
ctil[a1,y0], ntil[a1,y0] = nlsolve( x->euler!([1.0,1.0],x,y0,a1,cgrid),[.5,0.85]).zero
astar[a1,y0] = (ctil[a1,y0]+A[a1] -y[y0]*w*ntil[a1,y0])/R
end
end
itpc0 = LinearInterpolation(astar[:,y0],ctil[:,y0], extrapolation_bc=Line())
itpn0 = LinearInterpolation(astar[:,y0],ntil[:,y0], extrapolation_bc=Line())
for a1 =1:nA
if A[a1] >= astar[1,y0]
cgrid1[a1,y0] = itpc0(A[a1])
else
cgrid1[a1,y0] = R*A[a1]-A[1] +y[y0]*w*ntil[a1,y0]
end
end
end
distancec = maximum(abs.(cgrid1.-cgrid))
distancen = maximum(abs.(ngrid1.-ngrid))
cgrid1 = damp*cgrid1 .+ (1-damp)*cgrid

println("distance $(distance)")
end
itp = LinearInterpolation((A,y),cgrid1, extrapolation_bc=Line())
c(a,y) = itp(a,y)
k(a,y) = R*a+y-c(a,y)
#    return c,k,cgrid1
#end


@time c,cgrid1 =  EGM(A,y,pdfY;damp=1.0)

using Plots
plot(A,[A,R*A.+y[div(nY,2)].-c.(A,y[div(nY,2)])],label=["45","asset"])

plot(A,[c.(A,y[div(nY,2)])],label=["45","asset"])

=#
