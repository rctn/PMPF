% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Sample chain visualizer

addpath ../grbm/

clf;
dim=grbm.ImageDimensions;
H=grbm.nH;
V=grbm.nV;

nFigures=25;

% Image indeces,
nImgs=round((grbm.N-1)*rand(nFigures,1)+1);

for jFigure=1:nFigures
    iImg=nImgs(jFigure);
    
    Img=Samples_VbN(:,iImg);
    Img=Img(:);
    Img=bsxfun(@minus,Img,min(min(Img)));
    Img=bsxfun(@rdivide,Img,max(max(Img)));

    Red=reshape(Img(1:dim^2),dim,dim);
    Green=reshape(Img(dim^2+1:2*dim^2),dim,dim);
    Blue=reshape(Img(2*dim^2+1:3*dim^2),dim,dim);

    % Display in a subplot
    subplot(5,5,jFigure);
    imshow(cat(3,Red,Green,Blue));
    title(iImg);
    axis off;
end
suptitle('Samples in use 11^{th} Epoch');