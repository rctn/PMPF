% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Parallel Tempering Sampling
%     Obtains GBRBM samples using Parallel Tempering
%           
%     Steven Munn
%     email: stevenjlm@berkeley.edu

%     QUICK AND DIRTY VERSION -- NOT OPTIMIZED
%     Attempt 1 at optimization

function [Samples,AllChains]=sample_grbm_PT_beta_0611_1(Weights_HbV_ini, VBias_Vb1_ini, HBias_Hb1_ini,...
    Sigmas_Vb1_ini, grbm, AllChains)
    
    % Create a set of parameters
    Temperatures = linspace(0, 1, grbm.nPTChains);
    Weights_HbV=zeros(grbm.nH,grbm.nV,grbm.nPTChains);
    VBias_Vb1=zeros(grbm.nV,1,grbm.nPTChains);
    HBias_Hb1=zeros(grbm.nH,1,grbm.nPTChains);
    Sigmas_Vb1=zeros(grbm.nV,1,grbm.nPTChains);
    % For efficiency,
    IthMean_Vb1=grbm.ithMean_Vb1;
    IthSigma_Vb1=grbm.ithSigma_Vb1;
    % To avoid a complicateed mess of matrices, we use the parfor stucture
    % to execute these calculations which are independant of each other.
    temp=tic();
    parfor iPTChain=1:grbm.nPTChains
        Weights_HbV(:,:,iPTChain)=Temperatures(iPTChain)*Weights_HbV_ini;
        
        VBias_Vb1(:,iPTChain)=bsxfun(@plus,Temperatures(iPTChain)*VBias_Vb1_ini,...
            (1-Temperatures(iPTChain))*IthMean_Vb1);
        
        HBias_Hb1(:,iPTChain)=Temperatures(iPTChain)*HBias_Hb1_ini;
        
        Sigmas_Vb1(:,iPTChain)=sqrt(bsxfun(@plus,Temperatures(iPTChain)*Sigmas_Vb1_ini.^2,...
            (1-Temperatures(iPTChain))*(IthSigma_Vb1).^2));
    end
    temp=toc(temp);
    disp(temp);
    
    % Empty set of samples,
    Samples = zeros( grbm.nV, grbm.N );
    % Counts the number of swaps
    Swaps=0;
    % Initialize states in each grbm,
    if nargin < 6
        AllChains=normrnd( zeros(size(Weights_HbV_ini,2), grbm.nPTChains),...
            ones( size(Weights_HbV_ini,2), grbm.nPTChains));
    end
    
    % All hidden chains variable
    CurrentHidden=zeros( grbm.nH, grbm.nPTChains );
    
    
    % Start Chain
    % This is a time dependant chain; thus, the first for loop cannot be
    % a parallel loop.
    nSamplingSteps=grbm.N+grbm.BurnIn;
    for iSamplingStep=1:nSamplingSteps
        % Single Gibbs sampling over all the chains
        % This loops over independent chains, hence the parfor
        parfor iPTChain=1:grbm.nPTChains
            CurrentHidden(:,iPTChain) = sigmoid(bsxfun(@plus,...
                Weights_HbV(:,:,iPTChain) * bsxfun(@rdivide, AllChains(:,iPTChain), Sigmas_Vb1(:,iPTChain).^2),...
                HBias_Hb1(:,iPTChain)));
            
            CurrentHidden(:,iPTChain) = (CurrentHidden(:,iPTChain) > rand(size(CurrentHidden(:,iPTChain))));
            v_mean = bsxfun(@plus, (CurrentHidden(:,iPTChain)' * Weights_HbV(:,:,iPTChain))', VBias_Vb1(:,iPTChain));
            AllChains(:,iPTChain) = normrnd(v_mean, Sigmas_Vb1(:,iPTChain*ones(size(AllChains(:,iPTChain),2),1)));
        end
        
        % Once all the chains are updated we can check for swaping
        % Is this faster than going through them in a serial manner??
        for iPTChain=2:grbm.nPTChains
            Energy1=E_vectorized( Weights_HbV(:,:,iPTChain), HBias_Hb1(:,iPTChain),...
                VBias_Vb1(:,iPTChain), Sigmas_Vb1(:,iPTChain),...
                AllChains(:,iPTChain-1) );
            Energy2=E_vectorized( Weights_HbV(:,:,iPTChain-1), HBias_Hb1(:,iPTChain-1),...
                VBias_Vb1(:,iPTChain-1), Sigmas_Vb1(:,iPTChain-1),...
                AllChains(:,iPTChain) );
            Energy3=E_vectorized( Weights_HbV(:,:,iPTChain), HBias_Hb1(:,iPTChain),...
                VBias_Vb1(:,iPTChain), Sigmas_Vb1(:,iPTChain),...
                AllChains(:,iPTChain) );
            Energy4=E_vectorized( Weights_HbV(:,:,iPTChain-1), HBias_Hb1(:,iPTChain-1),...
                VBias_Vb1(:,iPTChain-1), Sigmas_Vb1(:,iPTChain-1),...
                AllChains(:,iPTChain-1) );
            ProbaSwap=min(1, exp(Energy1+Energy2-Energy3-Energy4));
            
            Swap = binornd(1, ProbaSwap);
            if Swap
                if iPTChain==grbm.nPTChains
                    Swaps=Swaps+1;
                end
                Index=1:grbm.nPTChains;
                Index(iPTChain)=Index(iPTChain)-1;
                Index(iPTChain-1)=Index(iPTChain-1)+1;
                AllChains=AllChains(:,Index);
            end
        end
        
        if iSamplingStep > grbm.BurnIn
            Samples(:,iSamplingStep-grbm.BurnIn)=AllChains(:,grbm.nPTChains);
        end
        
        %fprintf('\b\b\b\b\b%03.0f %%',iSamplingStep/nSamplingSteps*100);
    end
    %fprintf('\b\b\b\b\b');
    disp(Swaps);
        
end