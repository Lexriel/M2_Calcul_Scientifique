function [bro]=broyden(f,y0,h,eps,itermax)

d=length(y0);
A=zeros(d,d);

cpt=0;
test=1;

y=y0;

% On initialise la matrice A à l'aide des différences finies
% A(i,j) = [   f_i( x(1) , ... , x(j)+h , ... ,x(d) )
%            - f_i( x(1) , ... , x(j)-h , ... ,x(d) ) ] / (2h) .

for j=1:d
    y0p=y0;
    y0m=y0;
    y0p(j)=y0p(j)+h;
    y0m(j)=y0m(j)-h;
    
    A(:,j)=( f(y0p) - f(y0m) ) / (2*h);
end

while test
   r=y;
   y=y-A\f(y);
   u=f(y)-f(r); % r est le précédent y, bref r=y(n) et y=y(n+1).
   s=y-r;
   cpt=cpt+1;
   A=A+(u-A*s)*(s')/norm(s,2)^2;
   
   test=((norm(f(y),2)>eps)&&(cpt<itermax));    
    
end

bro=y;