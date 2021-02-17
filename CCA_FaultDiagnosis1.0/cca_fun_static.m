function [U, S, V, P,P_res, L,L_res] = cca_fun_static(X,Y)
    % CCA calculate canonical correlations
    %
    % [U, S, V, P, L] = cca(Y_tr,U_tr) where P and L contains the canonical correlation
    % matrices as columns and S is a matrix with corresponding canonical
    % correlations. The correlations are sorted in descending order. Y_tr and
    % U_tr are matrices where each column is a sample. Hence, Y_tr and U_tr must have
    % the same number of columns.
    %
    % Example: If Y is M*K and U_tr is L*K there are rank=MIN(M,L) solutions. U is
    % then M*M, V is L*L and S is M*L. P is M*rank, L is L*rank.
    %
    %
    % ? 2014 Zhiwen Chen, Duisburg-Essen universitet

    % --- Calculate covariance matrices ---

    [l,n_s]=size(Y);[m,~]=size(X);
    %%
    %CVA decomposition
    [U,S,V]=svd((X*X')^(-0.5)*(X*Y')*(Y*Y')^(-0.5));
    %%%%%%%%%%%%%%%%%%%%%%%% Determine the order 
    n_o = rank(S);
    S = S(1:n_o,1:n_o);
    % for i=1:size(S,2)
    %     if sum(sum(S(1:i,1:i)))/sum(sum(S))>0.8
    %         n_o=i;
    %         break
    %     end
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get L,P,X,Q,Omega,X
    P=(X*X')^(-0.5)*U(:,1:n_o);
    P_res = (X*X')^(-0.5)*U(:,n_o+1:end);
    L=(Y*Y')^(-0.5)*V(:,1:n_o);
    L_res=(Y*Y')^(-0.5)*V(:,n_o+1:end);
end