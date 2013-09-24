% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning
%     Data Energy
%     Calculates the Data energy

function Energy=E_vectorized( Weights, VBias, HBias, Sigmas, Data )
    % The function we implement here is,
    % $E\left(\mathbf{v}\right)=-{\displaystyle \sum_{j=1}^{H}\log\left[1+\exp\left(-{\displaystyle \sum_{i}\left(W_{ij}\frac{v_{i}}{\sigma_{i}}\right)-c_{j}}\right)\right]}-{\displaystyle \sum_{i=1}^{V}\frac{\left(v_{i}-b_{i}\right)^{2}}{2\sigma_{i}^{2}}}$

    % Now, step by step,
    % ${\bf a}=-{\displaystyle \sum_{i}\left(W_{ij}\frac{v_{i}}{\sigma_{i}}\right)-c_{j}}$
    Alpha=bsxfun(@plus,-Weights*(bsxfun(@rdivide,Data,Sigmas)),-HBias);

    % ${\bf b}=-{\displaystyle \sum_{i=1}^{V}\frac{\left(v_{i}-b_{i}\right)^{2}}{2\sigma_{i}^{2}}}$
    Beta=-sum(bsxfun(@rdivide,((bsxfun(@plus,Data,-VBias)).^2),(2*Sigmas.^2)));

    %We are now at,
    % $E\left(\mathbf{v}\right)=-{\displaystyle \sum_{j=1}^{H}\log\left[1+\exp\left({\bf a}\right)\right]}-{\displaystyle {\bf b}}$
    %But, first we calculate,
    % ${\bf g}=-{\displaystyle \sum_{j=1}^{H}\log\left[1+\exp\left({\bf a}\right)\right]}$
    Gamma=-sum(log(1+exp(Alpha)));
    %Now just add them,
    Energy=Gamma+Beta;
    %In one quick step,
    %E=-sum(log(1+exp(-Weights*(Data./Sigmas)-sum(HBias))))-sum(((Data-VBias).^2)./(2*Sigmas.^2));
end