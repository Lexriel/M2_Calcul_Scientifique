function [f]= fvinc(t,y)
[n,m]= size (y); f=zeros(n,m);
phix ='2*y(1) ';
phiy ='2*y(2) ';
phiz ='2*y(3) ';
H=2* eye (3);
mass =1; % Mass
F1='0*y(1) ';
F2='0*y(2) ';
F3='-mass *9.8 '; % Weight
xdot =zeros(3 ,1);
xdot (1:3)=y(4:6);
F=[ eval(F1); eval(F2); eval(F3)];
G=[ eval(phix); eval(phiy); eval(phiz)];
lambda=( mass*xdot'*H*xdot+F'*G)/(G'*G);
f(1:3)=y(4:6);
for k=1:3;
f(k+3)=(F(k)-lambda*G(k))/ mass ;
end
return