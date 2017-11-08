function [y] = burgers_exp3(dx,dt,visco,temps)

% FTCS explicit scheme, stable under some condition -> bad

nt=temps/dt;
t=linspace(0,temps,nt);


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

%Building tridigonal matrix

beta = - visco / (dx^2);
alpha = (2*visco / (dx^2)) + 1/dt;
gamma = -visco/(dx^2);

A = zeros(nx+1,nx+1);

for i=1:nx+1
    A(i,i) = alpha;
    A(i,i+1) = beta;
    A(i+1,i) = gamma;
end

T =A(2:nx,2:nx);


%algorithm

second_membre=zeros(nx-1);

for i=1:nt-1

y(i+1,1)=0;
y(i+1,nx+1)=0;    
    
%second membre

for j=2:nx
second_membre(j)=y(i,j)*( (y(i,j-1) - y(i,j))/dx + 1/dt );
end  
    

y(i+1,2:nx) = T\second_membre(2:nx)'; 

end

plot (x,y(nt,:),x,u);