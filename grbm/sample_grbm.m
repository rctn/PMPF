% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Model Sampler
%     Selects the proper sampling algorithm
%           gathers samples for learning

function [Samples, AllChains_VbNP,Energies_NPbSS]=sample_grbm(Weights_HbV, VBias_Vb1, HBias_Hb1,...
    Sigmas_Vb1, grbm, CurrentSamples)

    if grbm.ParallelTempering
        % Initialize states in each grbm,
        if grbm.debugSampler
            if nargin < 6
                AllChains_VbNP=normrnd( zeros(grbm.nV, grbm.nPTChains),...
                    ones( grbm.nV, grbm.nPTChains));
            else
                AllChains_VbNP=CurrentSamples;
            end
            [Samples, AllChains_VbNP,Energies_NPbSS]=sample_grbm_PT_beta_0611vec_debug(Weights_HbV, VBias_Vb1, HBias_Hb1,...
                                                 Sigmas_Vb1, grbm, AllChains_VbNP);
        else
            if nargin < 6
                AllChains_VbNP=normrnd( zeros(grbm.nV, grbm.nPTChains),...
                    ones( grbm.nV, grbm.nPTChains));
            else
                AllChains_VbNP=CurrentSamples;
            end
            [Samples, AllChains_VbNP]=sample_grbm_PT_beta_0611vec(Weights_HbV, VBias_Vb1, HBias_Hb1,...
                                                 Sigmas_Vb1, grbm, AllChains_VbNP);
        end
    else
        if grbm.SingleChain
            if nargin < 6
                CurrentSamples=normrnd( zeros(size(Weights_HbV,2), 1), ones( size(Weights_HbV,2), 1));
            end
            Samples=sample_grbm_single_chain(Weights_HbV, VBias_Vb1, HBias_Hb1,...
                                             Sigmas_Vb1, grbm, CurrentSamples);
            AllChains_VbNP=Samples(:,end);
        else
            if nargin < 6
                CurrentSamples=normrnd( zeros(size(Weights_HbV,2), grbm.BatchSize), ones( size(Weights_HbV,2), grbm.BatchSize));
            end
            Samples=sample_grbm_multichain(Weights_HbV, VBias_Vb1, HBias_Hb1,...
                                             Sigmas_Vb1, grbm, CurrentSamples);
            AllChains_VbNP=Samples;
        end
    end
end