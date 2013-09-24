% Reconstuction error estimate
HiddenUnitsOne_Hb1 = sigmoid(bsxfun(@plus,...
    Weights_HbV * bsxfun(@rdivide, Dall_VbN(:,150), Sigmas_Vb1.^2),...
    HBias_Hb1));
HiddenUnitsOne_Hb1 = (HiddenUnitsOne_Hb1 > rand(size(HiddenUnitsOne_Hb1)));
ReconMean_Vb1 = bsxfun(@plus, (HiddenUnitsOne_Hb1' * Weights_HbV)', VBias_Vb1);

dim=8;
figure;
Img=Dall_VbN(:,150);
Img=Img(:);
Img=bsxfun(@minus,Img,min(min(Img)));
Img=bsxfun(@rdivide,Img,max(max(Img)));

Red=reshape(Img(1:dim^2),dim,dim)';
Green=reshape(Img(dim^2+1:2*dim^2),dim,dim)';
Blue=reshape(Img(2*dim^2+1:3*dim^2),dim,dim)';

imshow(cat(3,Red,Green,Blue));
title(HUnitNo);
axis off;

figure;
Img=ReconMean_Vb1;
Img=Img(:);
Img=bsxfun(@minus,Img,min(min(Img)));
Img=bsxfun(@rdivide,Img,max(max(Img)));

Red=reshape(Img(1:dim^2),dim,dim)';
Green=reshape(Img(dim^2+1:2*dim^2),dim,dim)';
Blue=reshape(Img(2*dim^2+1:3*dim^2),dim,dim)';

imshow(cat(3,Red,Green,Blue));
title(HUnitNo);
axis off;