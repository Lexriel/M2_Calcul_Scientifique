function [ deriv ] = Derspect(n)
%Computes the derivation operator in Chebyshev Space
deriv=zeros(n+1);
for i=1:n+1
    ci=1;
    if ((i==1)|(i==n+1))
        ci=2;
    end
    for j=i+1:n+1
        if ~( ((j+i)/2) == (int16((j+i)/2) )) 
            deriv(i,j)=2*(j-1)/ci;
        end
    end
end
end

