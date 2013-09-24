% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Energy Partial Derivatives Calculator
%     Computes the rate of change of the energy with respect to
%           all of its variables

% OBSOLETE

function partials = partialsE_grbm3(D,W,vbias,hbias,sigmas,use_single_sigma) %CM4
    %time_p=tic();
    
    %Partial W
    alpha=bsxfun(@plus,W*(bsxfun(@rdivide,D,sigmas.^2)),hbias);
        %alpha is H by nsamples
    denom=bsxfun(@rdivide,1,1+exp(-alpha));
        %so is denom
    numer=-bsxfun(@rdivide,D,sigmas.^2);
        %numer is V by nsamples
        
    V=size(vbias,1);
    H=size(hbias,1);
    n_samples=size(D,2);
    %partialW=zeros(H,V,n_samples);

    %crazy, sick matrix rearrangement, trust me, it works somehow!!
%     numer=numer(:);
%     %numer=reshape(permute(repmat(numer(:),1,H),[2 1]),[H*V n_samples]);
%     numer=reshape(permute(numer(:, ones(H,1)),[2 1]),[H*V n_samples]);
    rowIdx = (1:V);
    colIdx = 1:n_samples;
    %denom2=repmat(denom,V,1);
    %denom2=denom([1:size(denom,1)]' * ones(1,V),[1:size(denom,2)]' * ones(1,1));
    %equivalent to denom2=repmat(denom,V,1); but slightly faster
    partialW=bsxfun(@times,numer(rowIdx(ones(H,1),:), colIdx),...
        denom((1:size(denom,1))' * ones(1,V),(1:size(denom,2))' * ones(1,1)));
    
    %Partial c
    partialc=-denom;
    
    %partial b
    partialb=bsxfun(@rdivide,bsxfun(@plus,vbias,-D),sigmas.^2);
    
    %partial Sigma
    res=W'*denom;
    term1=bsxfun(@times,res,bsxfun(@rdivide,2*D,sigmas.^3));
        % Is 1 by nsamples
    term2=bsxfun(@rdivide,bsxfun(@plus,D,-vbias).^2,sigmas.^3);
    partialSig=bsxfun(@plus,term1,-term2);
    if use_single_sigma
        partialSig=repmat(mean(partialSig),[V 1]);
    end
    
    
    partials=[partialW;reshape(partialc,H,n_samples);...
        reshape(partialb,V,n_samples);reshape(partialSig,V,n_samples)];
    
%     time_p=toc(time_p);
%     fprintf(2,'Partials comupted in %d seconds\n',time_p);

end