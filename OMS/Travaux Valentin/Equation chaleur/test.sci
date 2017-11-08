function [M]= expli(t,T,h,x)
    for i=1:x/h+1,  u_0(i,1)=(i-1)*h*(1-(i-1)*h), 
        end
    M=[u_0,zeros( x/h+1,T/t)]

l=t/h^2,

for j=1:T/t, 
    for i=2:x/h , M(i,j+1)=(1-2*l)*M(i,j)+ l*M(i-1,j)+l*M(i+1,j), end
end
M