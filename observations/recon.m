% Reconstuction error estimate

addpath ../grbm/

clf;
dim=grbm.ImageDimensions;
H=grbm.nH;
V=grbm.nV;

nFigures=25;

% Image indeces,
nImgs=round((grbm.N-1)*rand(nFigures,1)+1);

for jFigure=1:nFigures
    % Image index
    iImg=nImgs(jFigure);
    HiddenUnitsOne_Hb1 = sigmoid(bsxfun(@plus,...
        Weights_HbV * bsxfun(@rdivide, Dall_VbN(:,iImg), Sigmas_Vb1.^2),...
        HBias_Hb1));
    HiddenUnitsOne_Hb1 = (HiddenUnitsOne_Hb1 > rand(size(HiddenUnitsOne_Hb1)));
    ReconMean_Vb1 = bsxfun(@plus, (HiddenUnitsOne_Hb1' * Weights_HbV)', VBias_Vb1);
    
    Img=ReconMean_Vb1;
    % Nomralize the filter to values between 0 and 1
    Img=bsxfun(@minus,Img,min(min(Img)));
    Img=bsxfun(@rdivide,Img,max(max(Img)));

    % Get color channels
    Red=reshape(Img(1:dim^2),dim,dim)';
    Green=reshape(Img(dim^2+1:2*dim^2),dim,dim)';
    Blue=reshape(Img(2*dim^2+1:3*dim^2),dim,dim)';

    % Display in a subplot
    subplot(5,5,jFigure);
    imshow(cat(3,Red,Green,Blue));
    title(iImg);
    axis off;
end
suptitle('Reconstruction of 8 x 8 images');

figure;
for jFigure=1:nFigures
    iImg=nImgs(jFigure);
    Img=Dall_VbN(:,iImg);
    Img=Img(:);
    Img=bsxfun(@minus,Img,min(min(Img)));
    Img=bsxfun(@rdivide,Img,max(max(Img)));

    Red=reshape(Img(1:dim^2),dim,dim)';
    Green=reshape(Img(dim^2+1:2*dim^2),dim,dim)';
    Blue=reshape(Img(2*dim^2+1:3*dim^2),dim,dim)';

    % Display in a subplot
    subplot(5,5,jFigure);
    imshow(cat(3,Red,Green,Blue));
    title(iImg);
    axis off;
end
suptitle('Original 8 x 8 images');