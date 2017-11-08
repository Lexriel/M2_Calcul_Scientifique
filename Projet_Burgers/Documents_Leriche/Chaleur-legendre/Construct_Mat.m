function [M,H] = Construct_Mat(h,w,dPi,N)

M = diag(w);

H = zeros(size(M));

for i = 1:N+1
    for j = 1:N+1
        for m = 1:N+1
            H(i,j) = H(i,j) + w(m)*dPi(m,i)*dPi(m,j);
        end
    end
end

end