%=======================================================================
%
% Convert Radar Image to Probability map 
%
%=======================================================================

clear all;
close all;
clc;


R2D = 180 / pi ; D2R = pi / 180 ; 
YOK = 1 ; NOK = 0 ; NAN = nan ; 

FONTSIZE = 14 ; 


%-----------------------------------------------------------------------
% Main 
%-----------------------------------------------------------------------

tblImg = [1 : 5] ; 

for NumImg = tblImg

    strTitle    = sprintf("%06d", NumImg) ; 
    Pmap        = im2double(imread(strcat(strTitle, ".png"))) ; 

    azi = linspace(0, 360 * D2R , size(Pmap, 2)) ; 
    rng = linspace(0, 100, size(Pmap, 1)) ; 

    [AziDomain, RngDomain] = meshgrid(azi, rng) ; 

    xDomain = RngDomain .* cos(AziDomain) ; 
    yDomain = RngDomain .* sin(AziDomain) ; 

    if(YOK) 
        figure, box on, surf(AziDomain * R2D, RngDomain, Pmap), xlim([-inf, inf]) ; ylim([-inf, inf]) ; xlabel('azimuth [deg]'), ylabel('range [m]'), title('Probability map') ; set(gca, 'fontsize', FONTSIZE) 
        figure, box on, surf(yDomain, xDomain, Pmap) , xlim([-inf, inf]) ; ylim([-inf, inf]) ; xlabel('y [m]'), ylabel('x [m]'), title('Probability map') ; set(gca, 'fontsize', FONTSIZE) 
        figure, box on, surface(AziDomain * R2D, RngDomain, Pmap), xlim([-inf, inf]) ; ylim([-inf, inf]) ; xlabel('azimuth [deg]'), ylabel('range [m]'), title('Probability map') ; set(gca, 'fontsize', FONTSIZE) 
        figure, box on, surface(yDomain, xDomain, Pmap) , xlim([-inf, inf]) ; ylim([-inf, inf]) ; xlabel('y [m]'), ylabel('x [m]'), title('Probability map') ; set(gca, 'fontsize', FONTSIZE) 
        figure, box on, plot3(AziDomain * R2D, RngDomain, Pmap, '.'), xlim([-inf, inf]) ; ylim([-inf, inf]) ; xlabel('azimuth [deg]'), ylabel('range [m]'), title('Probability map') ; set(gca, 'fontsize', FONTSIZE) 
        figure, box on, plot3(yDomain, xDomain, Pmap,'.') , xlim([-inf, inf]) ; ylim([-inf, inf]) ; xlabel('y [m]'), ylabel('x [m]'), title('Probability map') ; set(gca, 'fontsize', FONTSIZE) 

    end

    save(strcat(strTitle, ".mat"), 'Pmap', 'xDomain', 'yDomain', 'RngDomain', 'AziDomain') ; 

end