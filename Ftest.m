function Ftest = Ftest(W,X,Y,order,numfeat)
    for ii=1:length(X)
        U = Umat2(X(ii,:),order);
        Yest(ii) = tmprod(W,U,(1:order));  % mode-n tensor-matrix product
    end
    Yest = Yest';
    Ftest=Y-Yest; % square error!!
end