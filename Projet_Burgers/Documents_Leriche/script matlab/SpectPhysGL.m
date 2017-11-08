function [ out ] = SpectPhysGL(n,out)
%Gives the operator of physic-spectral transform for Gauss
% Lobatto collocation points
for i=1:n+1
    for j=1:n+1
        out(i,j)=((-1)^(j-1))*cos((j-1)*(i-1)*pi/n);
    end
end
end

