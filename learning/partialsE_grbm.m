% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Energy Partial Derivatives Calculator
%     Computes the rate of change of the energy with respect to
%           all of its variables


function Partials = partialsE_grbm(Data,Weights,VBias,HBias,Sigmas,UseSingleSigma)
    %time_p=tic();
    
    % Wieghts partial derivatives
    Alpha=bsxfun(@plus,Weights*(bsxfun(@rdivide,Data,Sigmas.^2)),HBias);
        % Alpha is H by nSamples
    Denom=bsxfun(@rdivide,1,1+exp(-Alpha));
        % so is Denom
    Numerator=-bsxfun(@rdivide,Data,Sigmas.^2);
        % Numerator is V by nsamples
    
    % Get data sizes
    nV=size(VBias,1);
    nH=size(HBias,1);
    nSamples=size(Data,2);
    
    
    RowIndices = (1:nV);
    ColumnIndices = 1:nSamples;
    
    % Really need better explanation of what's going on here...
    PartialW=bsxfun(@times,Numerator(RowIndices(ones(nH,1),:), ColumnIndices),...
        Denom((1:size(Denom,1))' * ones(1,nV),(1:size(Denom,2))' * ones(1,1)));
    
    % Partial c -- patrial derivatives with repect to the hidden biases
    PartialC=-Denom;
    
    % Partial b -- patrial derivatives with repect to the visible biases
    PartialB=bsxfun(@rdivide,bsxfun(@plus,VBias,-Data),Sigmas.^2);
    
    % Partial Sigma
    res=Weights'*Denom;
    term1=bsxfun(@times,res,bsxfun(@rdivide,2*Data,Sigmas.^3));
        % Is 1 by nsamples
    term2=bsxfun(@rdivide,bsxfun(@plus,Data,-VBias).^2,Sigmas.^3);
    partialSig=bsxfun(@plus,term1,-term2);
    if UseSingleSigma
        partialSig=repmat(mean(partialSig),[nV 1]);
    end
    
    
    Partials=[PartialW;reshape(PartialC,nH,nSamples);...
        reshape(PartialB,nV,nSamples);reshape(partialSig,nV,nSamples)];
    
%     time_p=toc(time_p);
%     fprintf(2,'Partials comupted in %d seconds\n',time_p);

end