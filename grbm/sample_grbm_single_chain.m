% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Single Chain Gibbs Sampler
%     Obtains GBRBM samples using Gibbs sampling

function Samples=sample_grbm_single_chain(Weights_HbV, VBias_Vb1, HBias_Hb1,...
    Sigmas_Vb1, grbm, CurrentSamples)
    
    nSamplingSteps = grbm.BurnIn + 1 + (grbm.BatchSize-1)*(grbm.SpaceTweenSamples);

    Samples = zeros( size(Weights_HbV,2), grbm.BatchSize );
    % Next iteration number for which we will retain the samples,
    NextSampleNo = grbm.BurnIn+1;
    % Number of kept samples,
    iOut = 1;
    
    % Percentage progress report
    fprintf('%03.0f %%',0);
    
    for iSamplingStep=1:nSamplingSteps
        
        CurrentHidden_Hb1 = sigmoid(bsxfun(@plus, Weights_HbV * bsxfun(@rdivide, CurrentSamples, Sigmas_Vb1.^2), HBias_Hb1));
        CurrentHidden_Hb1 = (CurrentHidden_Hb1 > rand(size(CurrentHidden_Hb1)));
        %CurrentHidden_Hb1 = (CurrentHidden_Hb1 < rand(size(CurrentHidden_Hb1)));
        VisibleMean = bsxfun(@plus, (CurrentHidden_Hb1' * Weights_HbV)', VBias_Vb1);
        CurrentSamples = normrnd(VisibleMean, Sigmas_Vb1);
        %CM0508-1
        %CurrentSamples=VisibleMean;
%         t_mean(iSamplingStep)=VisibleMean;
        
        if iSamplingStep == NextSampleNo % copy to the output array if appropriate
            NextSampleNo = iSamplingStep + grbm.SpaceTweenSamples;
            Samples(:,iOut) = CurrentSamples;
            iOut = iOut + 1;
        end
        
        fprintf('\b\b\b\b\b%03.0f %%',iSamplingStep/nSamplingSteps*100);
    end
    fprintf('\b\b\b\b\b');
%     r=1;
%     disp(r);
end