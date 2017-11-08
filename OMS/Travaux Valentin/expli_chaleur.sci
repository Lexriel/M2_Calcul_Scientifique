function [M]= expli(t,T,h,x)
    for i=1:x/h-1,  u(i,1)=i*h*(1-i*h), 
    end
    M=[0;u;0]

    K=2*eye(x/h-1,x/h-1)
    for j=1:x/h-2,
         K(j,j+1)=-1,
         K(j+1,j)=-1,
    end
    
    for k=1:T/t,
        plot([0;u;0])
        halt()
        v=(eye(x/h-1,x/h-1)-t/h^2*K)*u
        M=[M [0;v;0]]
        u=v
    end
endfunction
