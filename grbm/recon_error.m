% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Reconstruction Error Estimate
%     Takes data and attempts to reconstruct it using the
%           GRBM parameters

function Error=recon_error( Weights_HbV, HBias_Hb1, VBias_Vb1, Sigmas_Vb1, Data_VbNB)

    %Reconstuction error estimate
    HiddenUnitsOne_Hb1 = sigmoid(bsxfun(@plus,...
        Weights_HbV * bsxfun(@rdivide, Data_VbNB, Sigmas_Vb1.^2),...
        HBias_Hb1));
    HiddenUnitsOne_Hb1 = (HiddenUnitsOne_Hb1 > rand(size(HiddenUnitsOne_Hb1)));
    ReconMean_VbNB = bsxfun(@plus, (HiddenUnitsOne_Hb1' * Weights_HbV)', VBias_Vb1);
    %recon = normrnd(recon_mean, repmat(Sigmas_Vb1,1,bsize));
    Error=mean(sum((ReconMean_VbNB - Data_VbNB).^2,2));
    
end