%=======================================================================
% Course : Mechatronics Integration Projects
% About : Convert Radar Image to Probability map 
% Author : Han,Kwanghee
% Initial Version : 03/16/22
% Modified : 04/02/22 
%=======================================================================
clear all; close all; clc;
colormap jet;

R2D = 180 / pi ; D2R = pi / 180 ; 
YOK = 0 ; NOK = 0 ; NAN = nan ; 

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
        pcolor(AziDomain * R2D, RngDomain, Pmap);
        pcolor(yDomain, xDomain, Pmap);
        figure, box on, surface(AziDomain * R2D, RngDomain, Pmap), xlim([-inf, inf]) ; ylim([-inf, inf]) ; xlabel('azimuth [deg]'), ylabel('range [m]'), title('Probability map') ; set(gca, 'fontsize', FONTSIZE); zlabel('Probability density'); 
        figure, box on, surface(yDomain, xDomain, Pmap) , xlim([-inf, inf]) ; ylim([-inf, inf]) ; xlabel('y [m]'), ylabel('x [m]'), title('Probability map') ; set(gca, 'fontsize', FONTSIZE); zlabel('Probability density');
        colormap;
%         set(get(h,'YLabel'),'String','Relative Power (dB)')
    end
    save(strcat(strTitle, ".mat"), 'Pmap', 'xDomain', 'yDomain', 'RngDomain', 'AziDomain') ; 

end
%% Calculation process for Free Space loss propagation
% lambda = 0.05 k = 1 rmax = 100 max power loss= 88dB
close all; clc
lambda  = 0.05; % wavelength[m] % i -> measurement distance
H       = 1.97; % Mounting Height[m]
alpha   = 0.4;% FP Coefficient
beta    = 0.6;% FN Coefficient
pred_Pmap   = zeros(576,400);
row     = 1;
col     = 1;
predicted = zeros(576,1);

for idx = 1 : 400 % range


       % Fitting to poly4
        cur_rng     = rng(1,:)'; % x
        cur_pmap    = Pmap(:,idx); % y
        fitted= fit(cur_rng,cur_pmap,'poly4');

        for i = 1 :576
        predicted(i) = fitted.p1*cur_rng(i)^4+fitted.p2*cur_rng(i)^3 ...
        +fitted.p3*cur_rng(i)^2+fitted.p4*cur_rng(i)+fitted.p5;
        end

        pred_Pmap(:,idx) = predicted;

        
end
lossval = zeros(576,400);
% Tversky loss function
% True prob = Pmap % pred prob -> predPmap
for i = 1 : 576

    for j = 1 : 400
        TP      = Pmap(i,j)*pred_Pmap(i,j);
        FN      = Pmap(i,j)*(1-pred_Pmap(i,j));
        FP      = (1-Pmap(i,j))*pred_Pmap(i,j);
      lossval(i,j)   = (TP/(TP+alpha*FP+beta*FN));

    end
end

calb = Pmap+lossval;
revP = im2uint8(calb);
im   = imread('000005.png');
figure(1);
imshow(im);
figure(2)
imshow(revP)