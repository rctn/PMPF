% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Parallel Tempering Sampling
%     Obtains GBRBM samples using Parallel Tempering

%     Attempt 1 at optimization
% Many Matrix manipulations come from
% http://www.ee.columbia.edu/~marios/matlab/Matlab%20array%20manipulation%20tips%20and%20tricks.pdf

function [Samples,AllChains_VbNP,Energies_NPbSS]=sample_grbm_PT_beta_0611vec_debug(Weights_HbV_ini, VBias_Vb1_ini, HBias_Hb1_ini,...
    Sigmas_Vb1_ini, grbm, AllChains_VbNP)
    
    % Create a set temperatures
    Temperatures_NPb1 = linspace(0, 1, grbm.nPTChains)';
    
    % Initialise parameters
    % nPTChains is noted NP in variable names
    % H time nPTChaines is noted NC
    HIndex_Hb1 = (1:grbm.nH)'; % These will be used in place of repmat
    VIndex_Vb1 = (1:grbm.nV)';
    
    % Weights MATRIX preperation ============
    Weights_NCbV=Weights_HbV_ini( HIndex_Hb1(:,ones(grbm.nPTChains,1)), :);
    % Weights_NCbV has nPTChains weight matrices concatinated vertically
    % underneath it.
    TIndex_1bNP=1:grbm.nPTChains;
    TVal_HbNP=TIndex_1bNP( ones(grbm.nH,1), TIndex_1bNP);
    TVals_NCb1=TVal_HbNP(:);
    Weights_NCbV=bsxfun( @times, Weights_NCbV, Temperatures_NPb1(TVals_NCb1',1));
    % VBias MATRIX preperation =============
    VBias_VxNPb1=VBias_Vb1_ini( VIndex_Vb1(:,ones(grbm.nPTChains,1)), :); % Vertical repmat
    TVal_VbNP=TIndex_1bNP( ones(grbm.nV,1), TIndex_1bNP);
    TVals_VxNPb1=TVal_VbNP(:);
    VBias_VxNPb1=bsxfun( @times, VBias_VxNPb1, Temperatures_NPb1(TVals_VxNPb1',1));
    
    IthMean_Vb1=grbm.ithMean_Vb1;
    IthMean_VxNPb1=IthMean_Vb1( VIndex_Vb1(:,ones(grbm.nPTChains,1)), :); % Vertical repmat
    
    VBias_VxNPb1=bsxfun( @plus, VBias_VxNPb1, (1-Temperatures_NPb1(TVals_VxNPb1',1)).*IthMean_VxNPb1);
    % HBias MATRIX preperation =============
    HBias_NCb1=HBias_Hb1_ini(HIndex_Hb1(:,ones(grbm.nPTChains,1)),:);
    HBias_NCb1=bsxfun( @times, HBias_NCb1, Temperatures_NPb1(TVals_NCb1',1));
    % Sigmas MATRIX preperation =============
    Sigmas_VxNPb1=Sigmas_Vb1_ini( VIndex_Vb1(:,ones(grbm.nPTChains,1)), :);
    Sigmas_VxNPb1=bsxfun( @times, Sigmas_VxNPb1.^2, Temperatures_NPb1(TVals_VxNPb1',1));
    
    IthSigma_Vb1=grbm.ithSigma_Vb1;
    IthSigma_VxNPb1=IthSigma_Vb1( VIndex_Vb1(:,ones(grbm.nPTChains,1)), :);
    
    Sigmas_VxNPb1=bsxfun( @plus, Sigmas_VxNPb1,...
        (1-Temperatures_NPb1(TVals_VxNPb1',1)).*IthSigma_VxNPb1.^2);
    
    
    % Empty set of samples,
    Samples = zeros( grbm.nV, grbm.N );
    Swaps=0;
    
    % Start Chain
    nSamplingSteps=grbm.N+grbm.BurnIn;
    Energies_NPbSS=zeros(grbm.nPTChains,nSamplingSteps);
    for iSamplingStep=1:nSamplingSteps
        
        % One step Gibbs sampling,
        
        % Division by sigmas, prep sigs,
        Sigmas_NPbV=permute(reshape(Sigmas_VxNPb1',grbm.nV,grbm.nPTChains),[2 1]);
        DataOverSigma_NPbV=bsxfun(@rdivide,AllChains_VbNP',Sigmas_NPbV);
        TIndex_NPb1=TIndex_1bNP';
        % Data over Sigma Index, intended to repmat the DataOverSigma_NPbV
        % matrix
        DoSIndex_HbNP=permute(TIndex_NPb1(:,ones(grbm.nH,1)),[2 1]);
        DataOverSigma_HxNPbV=DataOverSigma_NPbV(DoSIndex_HbNP(:)',:);
        FirstTerm_NCb1=sum(bsxfun( @times, Weights_NCbV, DataOverSigma_HxNPbV) , 2);
        InSigmoid_NCb1=bsxfun( @plus, FirstTerm_NCb1, HBias_NCb1);
        CurrentHidden_HbNP=reshape(sigmoid(InSigmoid_NCb1),grbm.nH,grbm.nPTChains);
        
        CurrentHidden_HbNP = (CurrentHidden_HbNP > rand(size(CurrentHidden_HbNP)));
        % Now to update the visible,
        
        % Transpose the sub-weight-matrices in Weights_NCbV
        p=grbm.nH;
        q=grbm.nV;
        m=grbm.nH*grbm.nPTChains;
        n=grbm.nV;
        Y = reshape( Weights_NCbV, [ p m/p q n/q ] );
        Y = permute( Y, [ 3 2 1 4 ] );
        WeightsTranspose_VxNPbH = reshape( Y, [ q*m/p p*n/q ] );
        clear Y;
        
        % Format hidden,
        CurrentHidden_HxNPb1=CurrentHidden_HbNP(:);
        CurrentHidden_HxNPbV=CurrentHidden_HxNPb1(:, ones(grbm.nV,1));
        Y = reshape( CurrentHidden_HxNPbV, [ p m/p q n/q ] );
        Y = permute( Y, [ 3 2 1 4 ] );
        CurrentHidden_VxNPbH = reshape( Y, [ q*m/p p*n/q ] );
        clear Y;
        
        % Gaussian mean calculation,
        FirstTerm_VxNPb1=sum(bsxfun( @times, WeightsTranspose_VxNPbH, CurrentHidden_VxNPbH) , 2);
        VMeans_VxNPb1=bsxfun(@plus,VBias_VxNPb1,FirstTerm_VxNPb1);
        AllChains_VbNP=reshape(normrnd(VMeans_VxNPb1,Sigmas_VxNPb1),grbm.nV,grbm.nPTChains);
        
        if ~mod(iSamplingStep/grbm.SwapInterval,1)
            % Assigning variables to 4-D arrays
            % Weights,
            p=grbm.nH; q=grbm.nV; m=grbm.nH*grbm.nPTChains; n=grbm.nV;
            Y = reshape( Weights_NCbV, [ p m/p q n/q ] );
            Weights_HbVbNPb1 = permute( Y, [ 1 3 2 4 ] );
            clear Y;
            % VBias
            p=grbm.nV; q=1; m=grbm.nV*grbm.nPTChains; n=1;
            Y = reshape( VBias_VxNPb1, [ p m/p q n/q ] );
            VBias_Vb1bNPb1 = permute( Y, [ 1 3 2 4 ] );
            clear Y;
            % HBias
            p=grbm.nH; q=1; m=grbm.nH*grbm.nPTChains; n=1;
            Y = reshape( HBias_NCb1, [ p m/p q n/q ] );
            HBias_Hb1bNPb1 = permute( Y, [ 1 3 2 4 ] );
            clear Y;
            % Sigmas
            p=grbm.nV; q=1; m=grbm.nV*grbm.nPTChains; n=1;
            Y = reshape( Sigmas_VxNPb1, [ p m/p q n/q ] );
            Sigmas_Vb1bNPb1 = permute( Y, [ 1 3 2 4 ] );
            clear Y;
            
            for iPTChain=2:grbm.nPTChains
                WeightsTemp1_HbV=Weights_HbVbNPb1(:,:,iPTChain,1);
                WeightsTemp2_HbV=Weights_HbVbNPb1(:,:,iPTChain-1,1);
                VBiasTemp1_Vb1=VBias_Vb1bNPb1(:,:,iPTChain,1);
                VBiasTemp2_Vb1=VBias_Vb1bNPb1(:,:,iPTChain-1,1);
                HBiasTemp1_Hb1=HBias_Hb1bNPb1(:,:,iPTChain,1);
                HBiasTemp2_Hb1=HBias_Hb1bNPb1(:,:,iPTChain-1,1);
                SigmasTemp1_Vb1=Sigmas_Vb1bNPb1(:,:,iPTChain,1);
                SigmasTemp2_Vb1=Sigmas_Vb1bNPb1(:,:,iPTChain-1,1);
                
                
                Energy1=E_vectorized( WeightsTemp1_HbV, VBiasTemp1_Vb1,...
                    HBiasTemp1_Hb1, SigmasTemp1_Vb1,...
                    AllChains_VbNP(:,iPTChain-1) );
                Energy2=E_vectorized( WeightsTemp2_HbV, VBiasTemp2_Vb1,...
                    HBiasTemp2_Hb1, SigmasTemp2_Vb1,...
                    AllChains_VbNP(:,iPTChain) );
                Energy3=E_vectorized( WeightsTemp1_HbV, VBiasTemp1_Vb1,...
                    HBiasTemp1_Hb1, SigmasTemp1_Vb1,...
                    AllChains_VbNP(:,iPTChain) );
                Energy4=E_vectorized( WeightsTemp2_HbV, VBiasTemp2_Vb1,...
                    HBiasTemp2_Hb1, SigmasTemp2_Vb1,...
                    AllChains_VbNP(:,iPTChain-1) );
                ProbaSwap=min(1, exp(Energy1+Energy2-Energy3-Energy4));
                
                Energies_NPbSS(iPTChain,iSamplingStep)=Energy3;
                if iPTChain == 2
                    Energies_NPbSS(1,iSamplingStep)=Energy4;
                end

                Swap = binornd(1, ProbaSwap);
                if Swap
                    if iPTChain==grbm.nPTChains
                        Swaps=Swaps+1;
                    end
                    Index=1:grbm.nPTChains;
                    Index(iPTChain)=Index(iPTChain)-1;
                    Index(iPTChain-1)=Index(iPTChain-1)+1;
                    AllChains_VbNP=AllChains_VbNP(:,Index);
                end
            end
        end
        
        if iSamplingStep>grbm.BurnIn
            Samples(:,iSamplingStep)=AllChains_VbNP(:,grbm.nPTChains);
        end
        
        fprintf('\b\b\b\b\b%03.0f %%',iSamplingStep/nSamplingSteps*100);
    end
    fprintf('\b\b\b\b\b');
        
end