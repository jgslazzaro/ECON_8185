clear
syms k0 k1 h z
syms theta phi delta kss hss zss beta

hss =1;

u(k0,k1,h,z)=log(k0^theta *(z*h)^(1-theta) +(1-delta)*k0 -k1) + phi * log(1-h);

foc1(k0,k1,h,z)= beta*diff(u,k0);
foc2(k0,k1,h,z)= beta^2*diff(u,k1);
foc3(k0,k1,h,z)= diff(u,h);

eq1 = foc1(kss,kss,hss,1) + foc2(kss,kss,hss,1) == 0;
%eq2 = foc3(kss,kss,hss,1) == 0;

eqs = [eq1]%, eq2];
vars = [kss]%,hss];

SS = solve(eqs,vars,  'ReturnConditions', true);


