% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Theta Parameter Parsing
%     Extracts the different values in the Theta variable from MinFunc

function [ Weights_HbV, VBias_Vb1, HBias_Hb1, Sigmas_Vb1]= theta_parser( theta, grbm)

    % For shorthand,
    nH = grbm.nH;
    nV = grbm.nV;
    % Parse the parameters
    Weights_HxVb1 = theta(1:nH*nV);
    HBias_Hb1 = theta(nH*nV+1:nH*nV+nH);
    VBias_Vb1 = theta(nH*nV+nH+1:nH*nV+nV+nH);
    Sigmas_Vb1 = theta(nH*nV+nV+nH+1:nH*nV+2*nV+nH);
    
    Weights_HbV = reshape( Weights_HxVb1, nH, nV);