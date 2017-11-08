function Q = Connectivity_Matrix(Nel,N)

Q = zeros(Nel*(N+1),Nel*(N+1)-(Nel-1));

for i = 1:Nel
    j = i-1;
    np = N+1;
    
    if(i == 1)
        Q(1:np,1:np) = eye(np);
    elseif(i == 2)
        Q(j*np+1:i*np,j*np:i*np-1) = eye(np);
    else
        Q(j*np+1:i*np,j*np-(i-2):i*np-(i-1)) = eye(np);
    end
    
end

end