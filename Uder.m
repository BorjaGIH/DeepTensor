function Uder = Uder(X, indx)
    Uder = 1;
    X = [1,X];
    for ii=1:length(indx)
        Uder = Uder * X(indx{ii});
    end
end