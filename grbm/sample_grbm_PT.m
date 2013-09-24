% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Parallel Tempering Sampling
%     Obtains GBRBM samples using Parallel Tempering

%     QUICK AND DIRTY VERSION -- NOT OPTIMIZED

function Samples=sample_grbm_PT(Weights_HbV_ini, VBias_Vb1_ini, HBias_Hb1_ini,...
    Sigmas_Vb1_ini, grbm)
    
    % Create a set of parameters
    Temperatures = linspace(0, 1, grbm.nPTChains);
    for iPTChain=1:grbm.nPTChains
        Weights_HbV{iPTChain}=Temperatures(iPTChain)*Weights_HbV_ini;
        
        VBias_Vb1{iPTChain}=bsxfun(@plus,Temperatures(iPTChain)*VBias_Vb1_ini,...
            (1-Temperatures(iPTChain))*grbm.ithMean_Vb1);
        
        HBias_Hb1{iPTChain}=Temperatures(iPTChain)*HBias_Hb1_ini;
        
        Sigmas_Vb1{iPTChain}=sqrt(bsxfun(@plus,Temperatures(iPTChain)*Sigmas_Vb1_ini.^2,...
            (1-Temperatures(iPTChain))*(grbm.ithSigma_Vb1).^2));
    end
    
    % Empty set of samples,
    Samples = zeros( size(Weights_HbV_ini,2), grbm.N );
    % Initialize states in each grbm,
    CurrentSamples=normrnd( zeros(size(W,2), nPTChains),...
        ones( size(W,2), nPTChains));
    
    
    % Start Chain
    nSamplingSteps=grbm.N;
    for iSamplingStep=1:nSamplingSteps
        for iPTChain=1:grbm.nPTChains
            CurrentHidden(:,iPTChain) = sigmoid(bsxfun(@plus,...
                Weights_HbV{iPTChain} * bsxfun(@rdivide, CurrentSamples(:,iPTChain), Sigmas_Vb1{iPTChain}.^2),...
                HBias_Hb1{iPTChain}));
            
            CurrentHidden(:,iPTChain) = (CurrentHidden(:,iPTChain) > rand(size(CurrentHidden(:,iPTChain))));
            v_mean = bsxfun(@plus, (CurrentHidden(:,iPTChain)' * Weights_HbV{iPTChain})', VBias_Vb1{iPTChain});
            CurrentSamples(:,iPTChain) = normrnd(v_mean, Sigmas_Vb1{iPTChain}(:,ones(size(v,2),1)));
        end
        
        for iPTChain=2:grbm.nPTChains
            Energy1=E_vectorized( Weights_HbV{iPTChain}, HBias_Hb1{iPTChain},...
                VBias_Vb1{iPTChain}, Sigmas_Vb1{iPTChain},...
                CurrentSamples(:,iPTChain-1) );
            Energy2=E_vectorized( Weights_HbV{iPTChain-1}, HBias_Hb1{iPTChain-1},...
                VBias_Vb1{iPTChain-1}, Sigmas_Vb1{iPTChain-1},...
                CurrentSamples(:,iPTChain) );
            Energy3=E_vectorized( Weights_HbV{iPTChain}, HBias_Hb1{iPTChain},...
                VBias_Vb1{iPTChain}, Sigmas_Vb1{iPTChain},...
                CurrentSamples(:,iPTChain) );
            Energy4=E_vectorized( Weights_HbV{iPTChain-1}, HBias_Hb1{iPTChain-1},...
                VBias_Vb1{iPTChain-1}, Sigmas_Vb1{iPTChain-1},...
                CurrentSamples(:,iPTChain-1) );
            ProbaSwap=min(1, exp(Energy1+Energy2-Energy3-Energy4));
            
            Swap = binornd(1, ProbaSwap);
            if Swap
                Index=1:grbm.nPTChains;
                Index(iPTChain)=iPTChain-1;
                Index(iPTChain-1)=iPTChain+1;
                CurrentSamples=CurrentSamples(:,Index);
            end
        end
        
        Samples(iSamplingStep)=CurrentSamples(:,grbm.nPTChains);
        
        fprintf('\b\b\b\b\b%03.0f %%',iSamplingStep/nSamplingSteps*100);
    end
    fprintf('\b\b\b\b\b');
        
end