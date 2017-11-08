function [ out ] = PhySpectGL(n,out)
%Gives the operator of physic-spectral transform for Gauss
% Lobatto collocation points
for i=1:n+1
    ci=1;
    if((i==1)||(i==n+1))
        ci=2;
    end
    for j=1:n+1
        cj=1;
        if((j==1)||(j==n+1))
        cj=2;
        end
        out(i,j)=((-1)^(i-1))*2*cos((j-1)*(i-1)*pi/n)/(n*ci*cj);
    end
end
end

