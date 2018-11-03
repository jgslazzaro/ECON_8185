clear
syms k0 k1 h z
syms theta phi delta kss hss zss

kss =2
hss=3
zss =1
u(k0,k1,h,z)=log(k0^theta *(z*h)^(1-theta) +(1-delta)*k0 -k1) + phi * log(1-h);


g = gradient(u, [k0,k1,h,z]);
H = hessian(u, [k0,k1,h,z]);
d(k0,k1,h,z)=[k0-kss;k1-kss;h-hss; z-zss];


LQu = d' * g(kss,kss,hss,zss) +d' * H(kss,kss,hss,zss) * d