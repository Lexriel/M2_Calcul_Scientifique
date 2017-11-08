function [x,w,Pi,dPi] = Legendre(N)

[x,w,P] = lglnodes(N);

[o,t] = sort(x);
x = x(t);
w = w(t);
P = P(t,:);

dP = zeros(size(P));
dP(:,2) = 1;
for i = 3:length(dP)
    k = i-1;
    dP(:,k+1) = (2*k+1) * P(:,k) + dP(:,k-1);
end

%+++++ Construction des fonctions de base +++++%

Pi = zeros(size(P));
dPi = Pi;

for i = 1:N+1
    A = -(1-x.^2) .* dP(:,end);
    B = N*(N+1) * (x-x(i)) .* P(i,end);
    Pi(:,i) = A./B;
    Pi(i,i) = 1;
end

for i = 1:N+1
    for j = 1:N+1
        if(i == j)
            dPi(i,j) = 0;
        else
            dPi(i,j) = P(i,end)/(P(j,end)*(x(i)-x(j)));
        end
    end
end

dPi(1,1) = -N*(N+1)/4;
dPi(end,end) = N*(N+1)/4;
end