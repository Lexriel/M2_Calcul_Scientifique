%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                                                    %%%%%
%%%%%     RESOLUTION DE L'EQUATION DE LA CHALEUR PAR     %%%%%
%%%%%          LA METHODE DES ELEMENTS SPECTRAUX         %%%%%
%%%%%                                                    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;

%%%%%%%%%%%%%%%%
%% PARAMETRES %%
%%%%%%%%%%%%%%%%

NELT  = 1;
N     = 32;
x_in  = -1;
x_out = 1;
h     = (x_out-x_in)/NELT;

U_IN = 0; U_OUT = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CREATION DES MATRICES DE MASSE ET RAIDEUR %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%+++++ MATRICE DE CONNECTIVITE BOOLEENNE +++++%

Q = Connectivity_Matrix(NELT,N); I = eye(NELT);

%+++++ CALCUL DES POINTS GLL +++++%

[e,w,Pi,dPi] = Legendre(N);
X = []; f = [];
for i = 1:NELT
    x_1 = (i-1)*h + x_in;
    x_2 = i*h + x_in;
    
    x_e = (1-e)/2 * x_1 + (1+e)/2 * x_2;
    X = [X;x_e];
    f_e = -sin(pi*x_e);
    f = [f; f_e];
end
ind = [];
for i = 1:NELT-1
    ind = [ind i*(N+1)];
end
X(ind) = [];

%+++++ CONSTRUCTION DU PROBLEME ELEMENTAIRE +++++%

[m,k] = Construct_Mat(h,w,dPi,N);
k = 2/h * k; m = h/2 * m;

%+++++ CONSTRUCTION DU PROBLEME GLOBALE +++++%

M = Q'*kron(I,m)*Q;
K = Q'*kron(I,k)*Q;
F = Q'*kron(I,m)*f;

%+++++ IMPOSITION DES CONDITIONS AUX LIMITES PAR UNE +++++%
   %+++++ METHODE DES MULTIPLICATEURS DE LAGRANGE +++++%

E = zeros(size(M,1)+2,size(M,2)+2);
A12 = zeros(size(K,2),2);
A12(1,1) = 1; A12(end,2) = 1;
E(1:length(M),1:length(M)) = M;
A = [-K A12;A12' zeros(2,2)];

E11 = M; A11 = -K;

F_ext = [F ; U_IN ; U_OUT];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% REPONSE A UN FORCAGE STATIONNAIRE %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U_ext = A\F_ext; U = U_ext(1:end-2);
plot(X,U);
