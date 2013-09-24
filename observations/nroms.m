% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Min and Max L2 Norm filter Visualizer
%     Computes the L2 norms of all the hidden unit filters and
%           displays them in 2 windows

addpath ../grbm/

clf;
dim=grbm.ImageDimensions;
H=grbm.nH;
V=grbm.nV;
Norms=zeros(H,1);
%[Weights,VBias,HBias,Sigmas]=theta_parser(Theta,grbm);
Weights=Weights_HbV;

% Compute the L2 norms of all the hidden unit filters
for k=1:H
    Norms(k)=norm(Weights(k,:));
end

% Find the indices of the largest and smallest filters
nFigures=25;
HUnitsMaxNorm=zeros(nFigures,1);
HUnitsMinNorm=zeros(nFigures,1);
for iFigure=1:nFigures
    [~,HUnitsMaxNorm(iFigure)]=max(Norms);
    [~,HUnitsMinNorm(iFigure)]=min(Norms);
    Norms(HUnitsMaxNorm(iFigure))=NaN;
    Norms(HUnitsMinNorm(iFigure))=NaN;
end

% Display the largest filters
for jFigure=1:nFigures
    % H unit index,
    HUnitNo=HUnitsMaxNorm(jFigure);
    % Get weights between that hidden unit and the visible ones
    Img=Weights(HUnitNo,:);
    % Img is now nV^2 by 1
    Img=Img(:);
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
    title(HUnitNo);
    axis off;
end
suptitle('Largest L2 filters for 8 x 8 images');

figure;
for jFigure=1:nFigures
    HUnitNo=HUnitsMinNorm(jFigure);
    Img=Weights(HUnitNo,:);
    Img=Img(:);
    Img=bsxfun(@minus,Img,min(min(Img)));
    Img=bsxfun(@rdivide,Img,max(max(Img)));

    Red=reshape(Img(1:dim^2),dim,dim)';
    Green=reshape(Img(dim^2+1:2*dim^2),dim,dim)';
    Blue=reshape(Img(2*dim^2+1:3*dim^2),dim,dim)';

    subplot(5,5,jFigure);
    imshow(cat(3,Red,Green,Blue));
    title(HUnitNo);
    axis off;
end
suptitle('Smallest L2 filters for 8 x 8 images');

% Cleanup
clear dim;
clear H;