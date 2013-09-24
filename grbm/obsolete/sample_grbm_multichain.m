%======================================================
%MPF Learning for natural Images
%Gaussian-Bernoulli Restricted Boltzmann Machine Sampler
%Adapted from
%
% grbm_sample - Gibbs sampler
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
%Steven Munn, Jascha Sohl-Dickstein
%Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
%email: stevenjlm@berkeley.edu
%======================================================

function v=sample_grbm_multichain(W, hbias, vbias, sigmas, n_samples, burnin,v)

    if nargin <7
        v=normrnd( zeros(size(W,2), n_samples), ones( size(W,2), n_samples));
    end
    
    for si=1:burnin
        h1 = sigmoid(bsxfun(@plus, W * bsxfun(@rdivide, v, sigmas.^2), hbias));
        h1 = (h1 > rand(size(h1)));
        v_mean = bsxfun(@plus, (h1' * W)', vbias);
        v = normrnd(v_mean, sigmas(:,ones(size(v,2),1)));
    end
    
end