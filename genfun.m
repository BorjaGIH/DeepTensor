function Y = genfun(X,order,numfeat)
    ex = randi(order,1,numfeat);
    el = randi(numfeat,1,numfeat);
    coef = randi(4,1,numfeat);
    for ii=1:size(X,1)
        Y(ii) = sum((X(ii,el).*coef).^ex);
    end
end