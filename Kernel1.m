classdef Kernel1 < TensorOptimizationKernel

properties
    x       % input data
    y       % label or output data
    lambda  % regularization parameter (when needed)
    numfeat % number of features 
    order   % order of the tensor
    rank    % rank of the tensor for decomposition (constraint/efficient representation)
end

methods

function this = Kernel1(x, y, lambda, numfeat, order, rank) % constructor
    this.x = x;
    this.y = y;
    this.lambda = lambda;
    this.numfeat = numfeat;
    this.order = order;
    this.rank = rank;
end

function fval = objfun(this,z) % objective function
    X = this.x;
    Y = this.y;
    indx = cell(1,this.order);
    for ii=1:length(X)
        for jj=1:numel(z) % warning! check correctness
            [indx{:}] = ind2sub(size(z),jj);
            indVec = cell2mat(indx);
            w2 = 0;
            for kk=1:this.rank
                w1 = 1;
                for ll=length(z)
                    w1 = w1 * z{ll}(indVec(ll),kk);
                end
                w2 = w2 + w1;
            end
            Y(ii) = w2 * prod(X(ii,indVec));
        end
    end
end 

function grad = grad(this,z) % column vector with the gradient
    %% analytic
    X = this.x;
    Y = this.y;
    
    
    
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
%     grad = TensorOptimizationKernel.serialize(deriv(@this.objfun, z, this.objfun(z), 'gradient'));
end

function isvalid = validate(this, z)
    isvalid = true; % you can ignore this for now
end

function initialize(this, z) % initialize some things, e.g., cached variables.
    
end 

end

end