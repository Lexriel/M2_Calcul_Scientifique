elements_finis [y] = elements_finis(nu, N, dt, temps)

% dt : pas en temps

% On cherche à résoudre A.U = B

A = sparse(N+1,N+1);
B = zeros(N+1,1);       % membre de droite
V = zeros(N+1,10001);   % vecteur représentant U~
h = 1/N;                % pas temporel
U = zeros(Nt,N+2);      % solution U à chaque temps t
Y = zeros(N+1);         % solution temporaire U à chaque étape temporelle


% On définit la fonction U0 à t = 0 connu
for i = 1:N+1
    U0(i) = sin(pi*(i-1)*h);
end

% On crée un tableau qui contiendra tous les résultats U à chaque étape, la dernière colonne de U est le temps
% on ne connait donc que sa dernière colonne (le temps), sa première ligne (U0), sa première et avant dernière colonne (conditions aux bords = 0)
Nt = ceil(temps/dt);
for i = 1:Nt
    U(i,N+2) = (i-1)*dt
end
for j = 1:N+1
    U(1,j) = U0(j);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%                            1ere étape : t = 0                            %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% On définit la fonction V = U0~ à t = 0 connu.
% Comme on doit calculer une intégrale de 0 à 1 par la suite, on la 
% discrétisera en une somme de 1000 éléments, pas = 1/1000.
for k = 1:N+1
    for j = 1:1001
        V(k,j) = sin(pi*((k-1)*h+(j-1)/1000)*h);   % (j-1)/1000 représente tous les x^ compris dans [0;1]
    end
end

% Calcul de la matrice d'assemblage locale
d1 = h/(6*dt);
d2 = h/(3*dt);
integrale1 = zeros(N+1);
integrale2 = zeros(N+1);
Aloc = zeros(2,2,N+1);

    % Définition des intégrales de 0 à 1 de U0~ et de U0~*x^
    for k = 1:N+1
        for j = 1:1001
            integrale1(k) = integrale1(k) + V(k,j)/1000;
            integrale2(k) = integrale2(k) + (j-1)/1000*V(k,j)/1000;
    	end
    end

        % Aloc^k est définit sur T_k
	for k = 1:N+1
	    for j = 1:1001
	        Aloc(1,1,k) = Aloc(1,1,k) + d2 - integrale1(k);
	        Aloc(1,2,k) = Aloc(1,2,k) + d1 + integrale1(k) - integrale2(k);
	        Aloc(2,1,k) = Aloc(2,1,k) + d1 - integrale1(k) - integrale2(k);
	        Aloc(2,2,k) = Aloc(2,2,k) + d2 + integrale2(k);                           % approximation de l'integrale de U⁰~ par la méthode de la moyenne
	    end
	end


% remplissage de la matrice A
% attention A(0,0) sur le projet correspond ici à A(1,1)
for k = 1:N+1
    A(k-1,k) = A(k-1,k,k) + Aloc(1,2,k);
    A(k,k-1) = A(k,k-1,k) + Aloc(2,1,k);
    A(k,k) = A(k,k,k) + Aloc(2,2,k);
    A(k-1,k-1) = A(k-1,k-1,k) + Aloc(1,1,k);
end

% calcul des intégrale sur[0;1] de d²U0~/dt² et d²U0~/dt²*x^
derivee2_integrale1 = zeros(N+1);
derivee2_integrale2 = zeros(N+1);

    % Définition des intégrales de 0 à 1 de U0~ et de U0~*x^
    % k = 1
    for j = 1:1001
        derivee_integrale1(1) = derivee_integrale1(1) + (V(2,j)-2*V(1,j))/(h*h*1000);
        derivee_integrale2(1) = derivee_integrale2(1) + (j-1)/1000*(V(2,j)-2*V(1,j))/(h*h*1000);
    end

    for k = 2:N
        for j = 1:1001
            derivee_integrale1(k) = derivee_integrale1(k) + (V(k+1,j)-2*V(k,j)+V(k-1,j))/(h*h*1000);
            derivee_integrale2(k) = derivee_integrale2(k) + (j-1)/1000*(V(k+1,j)-2*V(k,j)+V(k-1,j))/(h*h*1000);
    	end
    end

    % k = N+1
    for j = 1:1001
        derivee_integrale1(N+1) = derivee_integrale1(N+1) + (-2*V(N+1,j)+V(N,j))/(h*h*1000);
        derivee_integrale2(N+1) = derivee_integrale2(N+1) + (j-1)/1000*(-2*V(N+1,j)+V(N,j))/(h*h*1000);
    end


  
% remplissage du second membre B(k) = B_k = B_k^k + B_k^(k+1)
for k=2:N-1       % les bords restent à 0
    B(k) = B(k) + h/dt * integrale1(k) + nu/h * derivee2_integrale1(k); % B_k = B_k^k + B_k^(k+1)
end

% conditions aux limites
% B(1) = B(N+1) = 0 déjà effectué
A(1,:) = 0;
A(N+1,:) = 0;
A(1,1) = 1;
A(N+1,N+1) = 1;
B(2:N) = B(2:N) - A(1,2:N)*U(1);
B(2:N) = B(2:N) - A(N+1,2:N)*U(N+1);
A(2:N,0) = 0;
A(2:N,N+1) = 0;

% on résout A*U1 = B 
Y = A\B;
U(2,:) = Y(:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%                       boucle d'étapes : t = (m-1)*dt                     %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% On itère ce procédé en recalculant à chaque étape la matrice A et le vecteur B

for m = 3:Nt

% U0 représente ici U^m (=U^n) connu, on va chercher U^(m+1) (=U^(n+1))
for i = 1:N+1
    U0(:) = U(m-1,:);
end

% On définit la fonction V = U0~ à t = 0 connu.
% Comme on doit calculer une intégrale de 0 à 1 par la suite, on la 
% discrétisera en une somme de 1000 éléments, pas = 1/1000.
for k = 1:N+1
    for j = 1:1001
        V(k,j) = ??? ;
        % Je ne peux pas discrétiser V n'ayant pas une expression algébrique de U1, en effet je dois appliquer une transformation affine au tableau U1, ce qui n'est pas possible sans une expression algébrique, il doit exister un procédé que je ne connais pas pour l'obtenir
    end
end

% admettons qu'on aie V(k,j) pour poursuivre

% Calcul de la matrice d'assemblage locale
integrale1 = zeros(N+1);
integrale2 = zeros(N+1);
Aloc = zeros(2,2,N+1);

    % Définition des intégrales de 0 à 1 de U0~ et de U0~*x^
    for k = 1:N+1
        for j = 1:1001
            integrale1(k) = integrale1(k) + V(k,j)/1000;
            integrale2(k) = integrale2(k) + (j-1)/1000*V(k,j)/1000;
    	end
    end

        % Aloc^k est définit sur T_k
	for k = 1:N+1
	    for j = 1:1001
	        Aloc(1,1,k) = Aloc(1,1,k) + d2 - integrale1(k);
	        Aloc(1,2,k) = Aloc(1,2,k) + d1 + integrale1(k) - integrale2(k);
	        Aloc(2,1,k) = Aloc(2,1,k) + d1 - integrale1(k) - integrale2(k);
	        Aloc(2,2,k) = Aloc(2,2,k) + d2 + integrale2(k);                           % approximation de l'integrale de U⁰~ par la méthode de la moyenne
	    end
	end


% remplissage de la matrice A
% attention A(0,0) sur le projet correspond ici à A(1,1)
for k = 1:N+1
    A(k-1,k) = A(k-1,k,k) + Aloc(1,2,k);
    A(k,k-1) = A(k,k-1,k) + Aloc(2,1,k);
    A(k,k) = A(k,k,k) + Aloc(2,2,k);
    A(k-1,k-1) = A(k-1,k-1,k) + Aloc(1,1,k);
end

% calcul des intégrale sur[0;1] de d²U0~/dt² et d²U0~/dt²*x^
derivee2_integrale1 = zeros(N+1);
derivee2_integrale2 = zeros(N+1);

    % Définition des intégrales de 0 à 1 de U0~ et de U0~*x^
    % k = 1
    for j = 1:1001
        derivee_integrale1(1) = derivee_integrale1(1) + (V(2,j)-2*V(1,j))/(h*h*1000);
        derivee_integrale2(1) = derivee_integrale2(1) + (j-1)/1000*(V(2,j)-2*V(1,j))/(h*h*1000);
    end

    for k = 2:N
        for j = 1:1001
            derivee_integrale1(k) = derivee_integrale1(k) + (V(k+1,j)-2*V(k,j)+V(k-1,j))/(h*h*1000);
            derivee_integrale2(k) = derivee_integrale2(k) + (j-1)/1000*(V(k+1,j)-2*V(k,j)+V(k-1,j))/(h*h*1000);
    	end
    end

    % k = N+1
    for j = 1:1001
        derivee_integrale1(N+1) = derivee_integrale1(N+1) + (-2*V(N+1,j)+V(N,j))/(h*h*1000);
        derivee_integrale2(N+1) = derivee_integrale2(N+1) + (j-1)/1000*(-2*V(N+1,j)+V(N,j))/(h*h*1000);
    end
  
% remplissage du second membre B(k) = B_k = B_k^k + B_k^(k+1)
for k=2:N-1       % les bords restent à 0
    B(k) = B(k) + h/dt * integrale1(k) + nu/h * derivee2_integrale1(k); % B_k = B_k^k + B_k^(k+1);
end

% conditions aux limites
% B(1) = B(N+1) = 0 déjà effectué
A(1,:) = 0;
A(N+1,:) = 0;
A(1,1) = 1;
A(N+1,N+1) = 1;
B(2:N) = B(2:N) - A(1,2:N)*U(1);
B(2:N) = B(2:N) - A(N+1,2:N)*U(N+1);
A(2:N,0) = 0;
A(2:N,N+1) = 0;

% on résout A*U1 = B 
Y = zeros(N+1);
Y = A\B;
U(2,1:N+1) = Y(1:N+1);

end    % fin de la boucle temporelle m = 3:Nt


% il reste à ajouter une instruction de traçage de courbe
% U(:,1:N+1) en fonction de U(:,N+2)
