function [A] = coeffMatrix(numfeat, order, X)
    indx = cell(1,order);
    indVec = zeros((numfeat+1)^order,order);
%     X = [ones(size(X,1),1),X];
    A = zeros(size(X,1),(numfeat+1)^order);

%     for ii=1:numel(ful(U))   % alternative, check what is faster
    for ii=1:(numfeat+1)^order  % build tensor indices for the datapoints, ONCE. 
        [indx{:}] = ind2sub(repmat(numfeat+1,1,order),ii);
        indVec(ii,:) = cell2mat(indx);
    end
    
    for ii=1:size(X,1)
        xii = [1, X(ii,:)];
        for jj=1:size(indVec,1)
            A(ii,jj) = prod(xii(indVec(jj,:)));
        end
    end
end