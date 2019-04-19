include("functions.jl")



R, B1y,P,F1,B2,A2y,kss,hss,Wy =   LQ_distorted(params_calibrated,steadystates)

K = range(kss-2,stop=kss+2,length =300)

u1= zeros(300,2)
uf = copy(u1)



for i=1:300
    global u,u1
    u1[i,:] += -F1*[K[i],zss,τhss, τxss, gss,1]
    uf[i,:] = u1[i,:]/(sqrt(β)) - (R\Wy')*[K[i],zss,τhss, τxss, gss,1]
end


using Plots
plot(K,[K uf[:,1]],labels = ["45" "LQ"],legend = :bottomright, title = "LQ capital policy function")
savefig("figure1.png")
#plot(K,[uf[:,2]],labels = ["labor"],legend = :bottomright)
