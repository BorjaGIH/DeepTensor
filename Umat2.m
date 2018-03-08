function U = Umat2(X, order)
    U = cell(1,order);
    for jj=1:order
        U{jj} = [1, X]; % Add "bias" term
    end
end