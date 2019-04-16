#Author: João Lazzaro
#Main code used to solve HW 4
include("question1.jl")

include("question2.jl")


#######################################################################################
#Stuff for question 3
#FIRST RUN QUESTION 1 and 2!!!!!!!!

#Get the VFI objects, see HW1 or the file "run_VFI.jl" to adjust parameters.


include("HW1_stuff\\run_VFI.jl")


#Plotting the consumption policies
plot(k,[cnplot.(k),c.(k),contplot.(k),policy_c, LQ_policyc],label=["Finite Elements" "True Function" "Continuous time" "VFI (150 gridpoints)" "LQ"],legend=:bottomright)
savefig("question3.png")

#######################################################################################
#Stuff for question 4
#FIRST RUN THE STUFF FOR QUESTION 1 BUT NOT 2!!!!!!!!

include("question1.jl")
include("run_VFI.jl")
k_finiteelements=0.5*ones(10)
k_VFI_index = Int(floor(KVFI/4))*ones(Int,10)
true_k =  0.5*ones(10)

for i=2:10
    global k_finiteelements, k_VFI_index
    k_finiteelements[i] = polkplot(k_finiteelements[i-1])
    k_VFI_index[i] = policy_k[k_VFI_index[i-1]]
    true_k[i] = β*θ*A*true_k[i-1]^θ
end
c_VFI = policy_c[k_VFI_index]
c_finiteelements = cnplot.(k_finiteelements)
true_c = c.(true_k)


plot(1:10,[c_finiteelements,true_c,c_VFI],label=["Finite Elements" "True Function" "VFI (150 gridpoints)"],legend=:bottomright)
savefig("question4.png")
