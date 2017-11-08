function [y, dx, dt, s] = burgers(nx,nt,temps)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


y=zeros(nt,2*nx+1);

t=linspace(0,temps,nt);
dt=t(2)-t(1);

x=linspace(-2, 2, 2*nx+1);
dx=x(2)-x(1);

% fonction source u0

u0 = @(p) 1*(p>0);
% u0 = @(p) 1*(p<0);
% u0 = @(p) exp(-10*p^2);
bord1 = 0;
bord2 = 1;

% fonction f
f = @(p) p^2/2;


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
            a_jdemi = y(i,j); %( f(y(i,j+1)) - f(y(i,j)) ) / (y(i,j+1)-y(i,j));
            
            f_roe_plus_demi  = 1/2*( f( y(i,j) ) + f( y(i,j+1) ) - abs(a_jdemi)*( y(i,j+1) - y(i,j) ) ); 
            f_roe_moins_demi = 1/2*( bord1 + f( y(i,j) ) - abs(a_jdemi)*( y(i,j) - bord1 ) ); 
            
            y(i+1,j) = y(i,j) -s*( f_roe_plus_demi - f_roe_moins_demi ); % Roe 
  
        elseif (j == 2*nx+1)
            a_jdemi = y(i,j); %( f(bord2) - f(y(i,j)) ) / (bord2-y(i,j));
            
            f_roe_plus_demi  = 1/2*( f( y(i,j) ) + bord2 ) - abs(a_jdemi)*( bord2 - y(i,j) ); 
            f_roe_moins_demi = 1/2*( f( y(i,j-1) ) + f( y(i,j) ) - abs(a_jdemi)*( y(i,j) - y(i,j-1) ) ); 
            
            y(i+1,j) = y(i,j) -s*( f_roe_plus_demi - f_roe_moins_demi ); % Roe 
   
        else
            a_jdemi = y(i,j); %( f(y(i,j+1)) - f(y(i,j)) ) / (y(i,j+1)-y(i,j));
            
            f_roe_plus_demi  = 1/2*( f( y(i,j) ) + f( y(i,j+1) ) - abs(a_jdemi)*( y(i,j+1) - y(i,j) ) ); 
            f_roe_moins_demi = 1/2*( f( y(i,j-1) ) + f( y(i,j) ) - abs(a_jdemi)*( y(i,j) - y(i,j-1) ) ); 
            
            y(i+1,j) = y(i,j) -s*( f_roe_plus_demi - f_roe_moins_demi ); % Roe 
        end     
   
    end  

    
    
end

plot (x,y(nt,:),x,u);

dt
dx
s