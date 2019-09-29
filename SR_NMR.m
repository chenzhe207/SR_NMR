function [Z, E] = SR_NMR(Xtr, X, lambda, p, q)
% This routine solves the following optimization problem,
% min_{Z,E} sum(|Ei|_*)+lambda*|Z|_1
% s.t., Xtt = XtrZ+E

tol = 1e-6; %recErr
maxIter = 1e3;
rho = 1.1;
max_mu = 1e10;
mu = 0.1;
[d, m] = size(Xtr); %n denotes the number of training samples
[d, n] = size(X); %N denotes the number of all samples

%% Initializing optimization variables
% intialize
Z = zeros(m, n);
J = zeros(m, n);
E = sparse(d, n);
T1 = zeros(d, n);
T2 = zeros(m, n);

%% Start main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;
    
    Zk = Z;  
    Ek = E;
    Jk = J;
    %update Z
    Z = inv(eye(m) + Xtr' * Xtr) * (Xtr' * (X - Ek + mu\T1) + Jk - mu\T2);
    
    %updata J
    tempJ = Z + T2/mu;
   
    J=max(0, tempJ - lambda / mu) + min(0, tempJ + lambda / mu);
    
    %update E
    for i = 1 : n
       Xi = reshape(X(:, i), p, q);
       XtrZ = zeros(p, q);
       for j = 1 : m
           Xtemp = reshape(Xtr(:, j), p, q);
           XtrZ = XtrZ + Xtemp * Z(j, i);
       end
       Ti = reshape(T1(:, i), p, q);
       temp= Xi - XtrZ + Ti/mu;
       [U,sigma,V] = svd(temp,'econ');
       sigma = diag(sigma);
       svpE = length(find(sigma>1/mu));
       if svpE>=1
         sigma = sigma(1:svpE)-1/mu;
       else
         svpE = 1;
         sigma = 0;
       end
       tt = U(:,1:svpE) * diag(sigma) * V(:,1:svpE)';
       E(:,i) = tt(:);
   end
    
 %% convergence check  
    leq1 = X - Xtr * Z - E;
    leq2 = Z - J;
    
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    
    if stopC<tol || iter>=maxIter
        break;
    else
        T1 = T1 + mu * leq1;
        T2 = T2 + mu * leq2;  
        mu = min(max_mu, mu * rho);
    end
    if (iter==1 || mod(iter, 5 )==0 || stopC<tol)
            disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stopC,'%2.3e') ]);
    end
   
end

end