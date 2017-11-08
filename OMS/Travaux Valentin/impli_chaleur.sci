function [M]= impli(t,T,h,x)
    for i=1:x/h-1,  u(i,1)=i*h*(1-i*h), 
    end
    M=[0;u;0]

    A=2*eye(x/h-1,x/h-1)
    for j=1:x/h-2,
         A(j,j+1)=-1,
         A(j+1,j)=-1,
    end
    
    for k=1:T/t,
        plot([0;u;0])
        halt()
        v=inv(eye(x/h-1,x/h-1)+t/h^2*A)*u
        M=[M [0;v;0]]
        u=v
    end
endfunction

