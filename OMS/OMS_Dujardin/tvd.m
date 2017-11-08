function [y, dx, dt, s] = tvd(nx,nt,a,temps)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


y=zeros(nt,2*nx+1);

t=linspace(0,temps,nt);
dt=t(2)-t(1);

x=linspace(-2, 2, 2*nx+1);
dx=x(2)-x(1);

% fonction source u0

u0 = @(p) 1/2*(p>0)*(p<3) + 1/2*(p>1)*(p<3) + 1/2*(p>2)*(p<3);%exp(-10*p^2);

% minmod(x,y)

minmod = @(a,b) sign(a)*min(abs(a), abs(b))*(a*b>0);


% Initialisation du vecteur y de départ

for i=1:2*nx+1
    y(1,i) = u0(-2+(i-1)*dx);
end

u = y(1,:);

s = dt/dx;

%algorithm

for i=1:nt-1

    

    for j=1:2*nx+1
    
        if (j == 1)
            g2 = minmod (2*y(i,j) , y(i,j+1)-y(i,j));
            g1 = 0;
            
            y(i+1,j) = y(i,j) -a*s*y(i,j) + a*s*(a*s-1)*(g2 - g1);
            
        elseif (j == 2)
            g2 = minmod (2*(y(i,j)-y(i,j-1)) , y(i,j+1)-y(i,j));
            g1 = minmod (2*y(i,j-1) , y(i,j)-y(i,j-1));
           
            y(i+1,j) = y(i,j) -a*s*(y(i,j)-y(i,j-1)) + a*s*(a*s-1)*(g2 - g1);
   
        elseif (j == 2*nx+1)
            g2 = minmod (2*(y(i,j)-y(i,j-1)) , -y(i,j));
            g1 = minmod (2*(y(i,j-1)-y(i,j-2)) , y(i,j)-y(i,j-1));

            y(i+1,j) = y(i,j) -a*s*(y(i,j)-y(i,j-1)) + a*s*(a*s-1)*(g2 - g1);  
   
        else 
            g2 = minmod (2*(y(i,j)-y(i,j-1)) , y(i,j+1)-y(i,j));
            g1 = minmod (2*(y(i,j-1)-y(i,j-2)) , y(i,j)-y(i,j-1));
            
            y(i+1,j) = y(i,j) -a*s*(y(i,j)-y(i,j-1)) + a*s*(a*s-1)*(g2 - g1); %Lax-Wendroff 
        end     
   
    end  

    
    
end

plot (x,y(nt,:),x,u);

dt
dx
s