%% Bayes linear regression using automatic relevance determination
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% group: a group id vector
%        e.g. group = [1 1 1 2 2 2 3 3 3 4 4 4]' indicates that there are
%        4 group with 3 members in each
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = bardgroupreg(y,X,group)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N, P] = size(PHI);

group = [0;group(:)]; % account for bias
groupid = unique(group);
NG = length(groupid);

alphas = 2*ones(P, 1);
beta = 10;
w = ones(P,1);
d_w = Inf;
evidence = -Inf;
d_evidence = Inf;
maxit = 500;
stopeps = 1e-6;
maxvalue = 1e9;

% [v,d] = eig(PHI'*PHI);
% d = diag(d);
d = myeig(PHI);

i = 1;
while (d_evidence > stopeps) && (d_w > stopeps)  && (i < maxit)
    wold = w;
    evidenceold = evidence;
    
    %% Regarding the update of the hyperparameters, refer to Page 347-348
    %% of PRML
    
    % eliminate very large alphas to avoid precision problem of sigma
    index0 = find(alphas > maxvalue);
    index1 = setdiff(1:P, index0);
    if (length(index1) <= 0)
        disp('Optimization terminated due that all alphas are large.');
        break;
    end
    alphas1 = alphas(index1);
    PHI1 = PHI(:,index1);
    
    [N1,P1] = size(PHI1);
    if (P1>N1)
        Sigma1 = woodburyinv(diag(alphas1), PHI1', PHI1, (1/beta)*eye(N));
    else
        Sigma1 = (diag(alphas1) + beta*PHI1'*PHI1)^(-1);
    end
    
    diagSigma1 = diag(Sigma1);
    w1 = beta*Sigma1*PHI1'*y;
    w(index1) = w1;
    if(~isempty(index0)) w(index0) = 0; end
    
    gamma1 = 1 - alphas1.*diagSigma1;
    gamma = zeros(size(alphas));
    gamma(index1) = gamma1;
    
    rmse = sum((y-PHI*w).^2);
    
    for g = 1:NG
        index_ig = find(group == groupid(g));
        w_ig = w(index_ig);
        
        if norm(w_ig) == 0, continue; end
        
        gamma_ig = gamma(index_ig);
        alpha_ig = sum(gamma_ig)/(w_ig'*w_ig);
        alphas(index_ig) = alpha_ig;
    end
    beta  = (N-sum(gamma))/rmse;

    evidence = (1/2)*sum(log(alphas)) + (N/2)*log(beta) - ...
        (beta/2)*rmse - (1/2)*w'*diag(alphas)*w - ...
        (1/2)*sum(log((beta*d+alphas))) - (N/2)*log(2*pi);
    
%     if (evidence < evidenceold)
%         disp('Optimization terminated due to a decreasing evidence.');
%         w = wold;
%         break;
%     end
    
    d_w = norm(w-wold);
    d_evidence = abs(evidence-evidenceold);
    
    disp(['Iteration ' num2str(i)  ': evidence = ' num2str(evidence) ...
        ', wchange = ' num2str(d_w) ', rmse = ' num2str(rmse) ', beta = ' num2str(beta)]);
    i = i + 1;
end

if(i < maxit)
    fprintf('Optimization of alpha and beta successfull.\n');
else
    fprintf('Optimization terminated due to max iteration.\n');
end

b = w(2:P);
b0 = w(1);

if nargout == 1
    model.b = b;
    model.b0 = b0;
    varargout{1} = model;
elseif nargout == 2
    varargout{1} = b;
    varargout{2} = b0;
end
