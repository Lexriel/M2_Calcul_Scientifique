function [res] = exo6(m)

% m: mass
% y0: position initiale + vitesse initiale (vecteur de R⁶)
% tspan: intervalle d'étude

% initialisation
y=zeros(6,1);
y1=zeros(6,1);

% constantes
g=9.8;
F=[0;0;-m*g];

% opérateurs différentiels
grad=[2*y(1);2*y(2);2*y(3)];
H=2*eye(3);

% définition de lambda=L
x2=[y(4);y(5);y(6)];
L=( m* x2'*H*x2 + grad'*F) / ( norm(grad,2)^2 );

for j=1:3
    y1(j)=y(3+j);
    y1(3+j)=1/m*(F(j)-L*2*y(j));
end

res=@(y) y1;