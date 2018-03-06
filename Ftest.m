function Ftest = Ftest(W,X,Y,order)
    for ii=1:length(X)
%         U={[1,X(ii,1)],[1,X(ii,2)],[1,X(ii,3)],[1,X(ii,4)],[1,X(ii,5)]};  % matrices for the mode-n multiplication
        U = Umat(X(ii,:),order);
        Yest(ii) = tmprod(W,U,[1,2,3,4,5]);  % mode-n tensor-matrix product
    end
    Yest = Yest';
    Ftest=Y-Yest;
end