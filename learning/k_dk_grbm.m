% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Objective Function and Derivative Calculator
%     Organizes various function to compute the objective function
%           K and its derviatives dK


function [K,dK]=k_dk_grbm( Theta, Xin_VbNB, TrainingData_VbNB, ThetaOld, grbm )

    % For mathemtical descriptions of this code, check equations.pdf
    
    nSegments=size(Xin_VbNB,2) / grbm.SegmentSz;
    if mod(nSegments,1)
        error('Segment size needs to be a power of 2');
    end
    
    % Parse the parameters
    [Weights,VBias,HBias,Sigmas]=theta_parser( Theta, grbm);
    [WeightsP,VBiasP,HBiasP,SigmasP]=theta_parser( ThetaOld, grbm);

    % Initialize K and dK variables
    KOverDataTotal=0;
    KOverSamplesTotal=0;
    dKOverDataTotal=0;
    dKOverSamplesTotal=0;
    
    % Percentage progress reporter
    fprintf('%03.0f %%',0);
    
    for iSegment=1:nSegments
        % Extracting segments of Training data and sample data
        TrainingDataSegment=TrainingData_VbNB(:,1+grbm.SegmentSz*(iSegment-1):grbm.SegmentSz+grbm.SegmentSz*(iSegment-1));
        XinSegment=Xin_VbNB(:,1+grbm.SegmentSz*(iSegment-1):grbm.SegmentSz+grbm.SegmentSz*(iSegment-1));
        
        % First factor of K with training data,
        EnergyDiffData=e_diff(TrainingDataSegment,Weights,VBias,HBias,Sigmas,WeightsP,VBiasP,HBiasP,SigmasP);
        KOverDataTotal=KOverDataTotal+1/size(TrainingData_VbNB,2)*sum(exp(1/2*(EnergyDiffData)));
        % SECOND FACTOR of K with sampled data from model,
        EnergyDiffSamples=e_diff(XinSegment,Weights,VBias,HBias,Sigmas,WeightsP,VBiasP,HBiasP,SigmasP);
        KOverSamplesTotal=KOverSamplesTotal+1/size(Xin_VbNB,2)*sum(exp(-1/2*(EnergyDiffSamples)));
        
        % dK
        PartialsE_overData=partialsE_grbm(TrainingDataSegment,Weights,VBias,HBias,Sigmas,grbm.UseSingleSigma);
        PartialsE_overSamples=partialsE_grbm(XinSegment,Weights,VBias,HBias,Sigmas,grbm.UseSingleSigma);

        % cumulators
        dKOverDataTotal=dKOverDataTotal+sum(bsxfun(@times,exp(1/2*(EnergyDiffData)),PartialsE_overData),2);
        dKOverSamplesTotal=dKOverSamplesTotal+sum(bsxfun(@times,exp(-1/2*(EnergyDiffSamples)),PartialsE_overSamples),2);
        
        % Percentage progress reporter
        fprintf('\b\b\b\b\b%03.0f %%',iSegment/nSegments*100);
    end
    % Eliminate percentage progress reporter
    fprintf('\b\b\b\b\b');
    
    % Objective function value
    K=KOverSamplesTotal*KOverDataTotal;
    
    % Terms in the objective function derivatives
    term1=+1/2*1/size(TrainingData_VbNB,2)*bsxfun(@times,dKOverDataTotal,KOverSamplesTotal);
    term2=-1/2*1/size(Xin_VbNB,2)*bsxfun(@times,dKOverSamplesTotal,KOverDataTotal);
    
    % Objective function derivatives
    dK=term1+term2;
    
    %This is to avoid overlearning!
    if K<0.5
        K=0;
        dK=zeros(size(dK,1),size(dK,2));
    end
end