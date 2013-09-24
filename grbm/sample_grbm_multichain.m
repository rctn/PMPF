% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Parallel Sampling
%     Samples each particle in parallel

function CurrentSamples=sample_grbm_multichain(Weights_HbV, VBias_Vb1, HBias_Hb1,...
                                             Sigmas_Vb1, grbm, CurrentSamples)

    for iStep=1:grbm.steps
        CurrentHidden_HbBS=sigmoid(bsxfun(@plus,...
            Weights_HbV * bsxfun(@rdivide, CurrentSamples, Sigmas_Vb1.^2),...
            HBias_Hb1));
        CurrentHidden_HbBS=(CurrentHidden_HbBS > rand(size(CurrentHidden_HbBS)));
        VisibleMean_VbBS = bsxfun(@plus, (CurrentHidden_HbBS' * Weights_HbV)', VBias_Vb1);
        CurrentSamples = normrnd(VisibleMean_VbBS, Sigmas_Vb1(:,ones(1,grbm.BatchSize)));
        fprintf('\b\b\b\b\b%03.0f %%',iStep/grbm.steps*100);
    end
    fprintf('\b\b\b\b\b');
end