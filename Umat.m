function U = Umat(X, order)
    U = cell(1,length(X));
    for jj=1:length(X)
        aux = [];
        for ii=0:order-1
            aux = [aux, X(jj)^ii];
        end
        U{jj}=aux;
    end
end