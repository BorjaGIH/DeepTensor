classdef Kernel1 < TensorOptimizationKernel

properties
    x       % input data
    y       % label or output data
    lambda  % regularization parameter (when needed)
    numfeat % number of features 
    order   % order of the tensor
    rank    % rank of the tensor for decomposition (constraint/efficient representation)
    % small change to check version control
end

methods

function this = Kernel1(x, y, numfeat, order, rank) % constructor
    this.x = x;
    this.y = y;
    this.numfeat = numfeat;
    this.order = order;
    this.rank = rank;
end

function fval = objfun(this,z) % objective function
    X = this.x;
    Y = this.y;
    Yest = zeros(length(X),1);

    % Construct matrix stacking rows
    for jj=1:this.rank
    b1 = [z{1}(:,jj)';zeros(this.order-1,this.numfeat+1)]; % block
    for ii=2:this.order
        b1 = [b1, [zeros(ii-1,this.numfeat+1);z{ii}(:,jj)';zeros(this.order-ii,this.numfeat+1)]]; % jj is the rank
    end
    if jj==1
        M1 = b1; % matrix (M1)
    else
        M1 = [M1;b1];
    end
    end    
    
    for ii=1:length(Yest) % loop through all datapoints
        xii = repmat(X(ii,:),1,this.order )';
        res = reshape(M1*xii,this.order,this.rank);
        Yest(ii) = sum(prod(res,1));
    end
    fval = (1/2)*sum((Y-Yest).^2); % no regularization
end 

function grad = grad(this,z) % column vector with the gradient
    %% analytic
%     X = this.x;
%     Y = this.y;   
% %     this.rank = rankest(z); % rank so that error is below threshold in CPD
%     [ucpd,~] = cpd(z,this.rank); % approximate tensor with fixed rank (constraint/efficient representation)
%     z = cpdgen(ucpd); % reconstruct tensor
%     for jj=1:numel(z)
%         prod1=zeros(length(X),1);
%         indx = cell(1,this.order);
%         [indx{:}] = ind2sub(size(z),jj);
%         for ii=1:length(X)
%             U = Umat2(X(ii,:),this.order);
%             Ud = Uder(X(ii,:),indx);
%             prod1(ii) = -(Y(ii) - tmprod(z,U,(1:this.order)))*Ud; % derivative 
%         end
%         grad(jj) = sum(prod1); % derivative
%     end

    %% numerical
    grad = TensorOptimizationKernel.serialize(deriv(@this.objfun, z, this.objfun(z), 'gradient'));
end

function isvalid = validate(this, z)
    isvalid = true; % you can ignore this for now
end

function initialize(this, z) % initialize some things, e.g., cached variables.
    
end 

end

end