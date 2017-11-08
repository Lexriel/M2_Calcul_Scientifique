function [x] = GaussLobatto(n)
%Computes the Gauss-Lobatto collocation points ordered from -1
% to 1
for i=1:n+1
    x(i)=-cos(pi*(i-1)/n);
end

