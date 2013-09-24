% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     PT Sample chains visualizer

addpath ../grbm/

clf;
dim=grbm.ImageDimensions;
H=grbm.nH;
V=grbm.nV;

nFigures=grbm.nPTChains;

for jFigure=1:nFigures
    
    Img=AllChains_VbNP(:,jFigure);
    Img=Img(:);
    Img=bsxfun(@minus,Img,min(min(Img)));
    Img=bsxfun(@rdivide,Img,max(max(Img)));

    Red=reshape(Img(1:dim^2),dim,dim)';
    Green=reshape(Img(dim^2+1:2*dim^2),dim,dim)';
    Blue=reshape(Img(2*dim^2+1:3*dim^2),dim,dim)';

    % Display in a subplot
    subplot(3,2,jFigure);
    imshow(cat(3,Red,Green,Blue));
    title(jFigure);
    axis off;
end
suptitle('Sample Chains');