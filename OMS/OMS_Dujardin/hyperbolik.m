function [y, dx, dt, s] = hyperbolik(nx,nt,a,temps)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


y=zeros(nt,2*nx+1);

t=linspace(0,temps,nt);
dt=t(2)-t(1);

x=linspace(-2, 2, 2*nx+1);
dx=x(2)-x(1);

%fonction source u0

u0 = @(p) exp(-10*p^2);

%Initialisation du vecteur y de départ

for i=1:2*nx+1
    y(1,i)=u0(-2+(i-1)*dx);
end

u=y(1,:);

%algorithm

for i=1:nt-1

    

    for j=1:2*nx+1
    
        if (j == 1)
               y(i+1,j)= y(i,j) -a/2*dt/dx*y(i,j+1) + a^2/2*(dt/dx)^2*(y(i,j+1) - 2*y(i,j));
   
        elseif (j == 2*nx+1)
               y(i+1,j)= y(i,j) -a/2*dt/dx*(-y(i,j-1)) + a^2/2*(dt/dx)^2*(-2*y(i,j) + y(i,j-1));
   
        else 
                y(i+1,j)= y(i,j) -a/2*dt/dx*(y(i,j+1)-y(i,j-1)) + a^2/2*(dt/dx)^2*(y(i,j+1) - 2*y(i,j) + y(i,j-1)); %Lax-Wendroff 
        end     
   
    end  

    
    
end

plot (x,y(nt,:),x,u);

dt
dx
s = dt/dx