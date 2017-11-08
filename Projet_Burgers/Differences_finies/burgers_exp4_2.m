function [y] = burgers_exp4_2(dx,dt,visco,temps)

% everything implicit

nt=temps/dt;
t=linspace(0,temps,nt);
maxi=0;

nx= 1/(dx);
x=linspace(0, 1, nx+1);

y=zeros(nt,nx+1);

%fonction source u0

u0 = @(p) sin(pi*p);

%Initialisation du vecteur y de départ

for i=1:nx+1
    y(1,i)=u0( (i-1)*dx);
end

u=y(1,:);


for i=1:nt-1

y(i+1,1)=0;
y(i+1,nx+1)=0;

%Building tridigonal matrix

beta = -visco / (dx^2);
alpha = (2*visco / (dx^2)) + 1/dt;
gamma = -visco/(dx^2);

A = zeros(nx+1,nx+1);

for j=1:nx+1
    A(j,j) = alpha + y(i,j)/dx;
    A(j,j+1) = beta;
    A(j+1,j) = gamma - y(i,j)/dx;   
end    

T =A(2:nx,2:nx);

%Building right-hand side

for j=2:nx
second_membre(j)=y(i,j)/dt;
end

y(i+1,2:nx)=T\second_membre(2:nx)';
derive(i+1) =( y(i+1,nx+1)-y(i+1,nx) )/dx;

if (derive(i+1) < maxi)
   maxi = derive(i);
   tmax=(i+1)*dt;
end
end

maxi
tmax
plot (t,derive);