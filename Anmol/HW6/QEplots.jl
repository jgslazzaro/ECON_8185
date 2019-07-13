include("QEfunctions.jl")

function log_utility(;β = 0.9,
                    ψ = 0.69,
                    Π = 0.5 * ones(2, 2),
                    G = [0.1, 0.2],
                    Θ = ones(2),
                    transfers = false)
    # Derivatives of utility function
    U(c,n) = log(c) + ψ * log(1 - n)
    Uc(c,n) = 1 ./ c
    Ucc(c,n) = -c.^(-2.0)
    Un(c,n) = -ψ ./ (1.0 .- n)
    Unn(c,n) = -ψ ./ (1.0 .- n).^2.0
    n_less_than_one = true
    return Model(β, Π, G, Θ, transfers,
                U, Uc, Ucc, Un, Unn, n_less_than_one)
end

function crra_utility(;
    β = 0.9,
    σ = 2.0,
    γ = 2.0,
    Π = 0.5 * ones(2, 2),
    G = [0.1, 0.2],
    Θ = ones(Float64, 2),
    transfers = false
    )
    function U(c, n)
        if σ == 1.0
            U = log(c)
        else
            U = (c.^(1.0 - σ) - 1.0) / (1.0 - σ)
        end
        return U - n.^(1 + γ) / (1 + γ)
    end
    # Derivatives of utility function
    Uc(c,n) =  c.^(-σ)
    Ucc(c,n) = -σ * c.^(-σ - 1.0)
    Un(c,n) = -n.^γ
    Unn(c,n) = -γ * n.^(γ - 1.0)
    n_less_than_one = false
    return Model(β, Π, G, Θ, transfers,
                U, Uc, Ucc, Un, Unn, n_less_than_one)
end

# Initialize μgrid for value function iteration
μgrid = range(-0.7, 0.01, length = 200)
log_example = log_utility()

log_example.transfers = false                             # Government can use transfers
#log_sequential = SequentialAllocation(log_example)       # Solve sequential problem
log_bellman = RecursiveAllocation(log_example, μgrid)    # Solve recursive problem

T = 20
sHist = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
G = [0.1,0.2]
#simulate
#sim_seq = simulate(log_sequential, 0.5, 1, T, sHist)
sim_bel = simulate(log_bellman, 0.5, 1, T, sHist)

#sim_seq_plot = hcat(sim_seq[1:3]...,
#            sim_seq[4], log_example.G[sHist], log_example.Θ[sHist] .* sim_seq[2])
sim_bel_plot = hcat(sim_bel[1:3]...,
            sim_bel[5], log_example.G[sHist], log_example.Θ[sHist] .* sim_bel[2])

#plot policies
titles = hcat("Consumption", "Labor Supply", "Government Debt",
              "Tax Rate", "Government Spending", "Output")
p = plot(size = (920, 750), layout = grid(3, 2),
         xaxis=(0:T), grid=false, titlefont=Plots.font("sans-serif", 10))
labels = fill(("", ""), 6)
labels[3] = ("Complete Market", "Incomplete Market")
plot!(p, title = titles)
for i = vcat(collect(1:4), 6)
#    plot!(p[i], sim_seq_plot[:, i], marker=:circle, color=:black, lab=labels[i][1])
    plot!(p[i], sim_bel_plot[:, i], marker=:utriangle, color=:blue, lab=labels[i][2],
          legend=:bottomright)
end
plot!()
plot!(p[5], G[sHist], marker=:circle, color=:blue, lab="")
