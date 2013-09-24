% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Energy Difference Calculator
%     Computes difference bewteen the calculated energy of a data set
%           using 2 different parameter sets

function EnergyDifference=e_diff( Data, Weights, VBias, HBias, Sigmas,...
    WeightsP, VBiasP, HBiasP, SigmasP)

    % For mathemtical descriptions of this code, check equations.pdf
    % Inside exponential,
            Alpha=Weights*bsxfun(@rdivide,Data,Sigmas.^2);
            Alpha=bsxfun(@plus,Alpha,HBias);

            AlphaP=WeightsP*bsxfun(@rdivide,Data,SigmasP.^2);
            AlphaP=bsxfun(@plus,AlphaP,HBiasP);

        % inside log,
        if max(max(Alpha))<700 && max(max(AlphaP))<700
            % Alpha and AlphaP are small enough for E_{diff} to
            % be computed exactly as described
            Gamma=bsxfun(@rdivide,1+exp(AlphaP),1+exp(Alpha));
        else
            % Alpha and AlphaP are large, we do not need to add 1 to
            % the quotient
            Gamma=exp(AlphaP-Alpha);
        end

        % First term, summation over the hiddent units
        FirstTerm=sum((log(Gamma)),1);
        
        % Second term, summation over the visible units
            Beta= bsxfun(@rdivide,(bsxfun(@plus,Data,-VBias)).^2,(2*Sigmas.^2));
            BetaP= bsxfun(@rdivide,(bsxfun(@plus,Data,-VBiasP)).^2,(2*SigmasP.^2));
        SecondTerm = sum((Beta-BetaP),1);
        
    EnergyDifference=FirstTerm+SecondTerm;
end