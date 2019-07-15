

#Consumption function comes from feasibility constraint:
consumption(n,g) = n - g

#The non dynamic part of the SP problem
function V(n,γ,b,g,μ)
    c = consumption(n,g)
    UC = Uc(c,n)
    UN = Un(c,n)
    V = U(c,n) +μ*b*UC+γ*(UC*(c-b)+n*UN)
    return V
end

#The optimal labor choice:
nstar(γ,b,g,μ) = optimize(n-> -V(n,γ,b,g,μ) ,g,1.0).minimizer
Vstar(γ,b,g,μ) = V(nstar(γ,b,g,μ),γ,b,g,μ)


#The expected Value function:
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



function SPFE(B,G,μ;nB=nB,nG=nG,nμ=nμ,β=β,Wgrid = Wgrid = ones(nB,nμ,nG))
    #= Inputs:
    Wgrid: Guess for value function
    B: Grid for debt
    G: Grid for government expenditures
    μ: Grid for the Lagrange Multiplier
    =#

    #= Outputs:
    Wgrid: Value Function
    policy: Policy functions
    =#

    #Preallocating stuff
    Wgrid1::Array{Float64,3}=copy(Wgrid) #to store updated values
    objective::Array{Float64,2} = ones(nμ,nB) #This will be the function min maxed
    policy::Array{Int64,4} = ones(Int64,nB,nμ,nG,2) #This will store the policy functions
    dist::Float64 =1.0 #Distance
    Vstargrid::Array{Float64,4} = ones(nB,nμ,nG,nμ) #This is the grid for the non0dynamic part of the SP problem
    EWgrid::Array{Float64,3} = ones(nB,nμ,nG) #This is the grid the expected value function
    #find the grid for the non dynamic part
    for γi=1:nμ,gi = 1:nG, μi = 1:nμ , bi = 1:nB
        Vstargrid[bi,μi,gi,γi] = Vstar(μ[γi],B[bi],G[gi],μ[μi])
    end

    while dist>1e-5
        #find the grid of the new guess, for each possible value of γ and b1
        for gi = 1:nG, γi = 1:nμ,b1 = 1:nB
            EWgrid[b1,γi,gi] = EW(B[b1],μ[γi],G[gi],Wgrid)
        end
            for gi = 1:nG,μi = 1:nμ ,bi = 1:nB
                #Joun the dynamic with non dynamic parts
                for γi=1:nμ, b1i=1:nB
                    objective[γi,b1i] = Vstargrid[bi,μi,gi,γi] +β * EWgrid[b1i,γi,gi]
                end
                ob2,indsB =findmax(objective,dims=2) #find the inner maximum of the SP problem
                Wgrid1[bi,μi,gi],indsμ = findmin(ob2) #find the outer minimum
                policy[bi,μi,gi,2] = indsμ[1] #get the policy indexes
                policy[bi,μi,gi,1] = indsB[indsμ[1]][2]
            end
        dist = maximum(abs.(Wgrid.-Wgrid1)) #chech convergence
        println("distance is $(dist)")
        Wgrid = 0.75*Wgrid1+0.25*Wgrid
    end
    return Wgrid,policy
end
