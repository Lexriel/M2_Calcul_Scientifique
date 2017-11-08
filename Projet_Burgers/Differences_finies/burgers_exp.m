function [y] = burgers_exp(dx,dt,visco,temps)

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

%algorithm

s= dt/dx;

for i=1:nt-1

    u(i,1)=0;
    u(i,nx+1)=0;
    
    for j=2:nx
            
        y(i+1,j)= dt*(       visco*( y(i,j+1) - 2*y(i,j) + y(i,j-1) )/(dx)^2    -   y(i,j)*( ( y(i,j) - y(i,j-1) )/(dx) )    ) + y(i,j);           
       
    end  

end

plot (x,y(nt,:),x,u);