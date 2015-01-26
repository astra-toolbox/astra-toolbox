function F = rebin_fan2par(RadonData, BetaDeg, D, thetaDeg)

%------------------------------------------------------------------------
% F = rebin_fan2par(RadonData, BetaDeg, D, thetaDeg)
%
% Deze functie zet fan beam data om naar parallelle data, door interpolatie
% (fast resorting algorithm, zie Kak en Slaney)
% Radondata zoals altijd: eerste coord gamma , de rijen
%                          tweede coord beta, de kolommen, beide hoeken in
%                          radialen
% PixPProj: aantal pixels per projectie (voor skyscan data typisch 1000)
% BetaDeg: vector met alle draaihoeken in graden
% D: afstand bron - rotatiecentrum in pixels, dus afstand
% bron-rotatiecentrum(um) gedeeld door image pixel size (um).
% thetaDeg: vector met gewenste sinogramwaarden voor theta in graden
%       de range van thetaDeg moet steeds kleiner zijn dan die van betadeg
% D,gamma,beta, theta zoals gebruikt in Kak & Slaney
%------------------------------------------------------------------------
%------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%------------------------------------------------------------------------
% $Id$

NpixPProj = size(RadonData,1);  % aantal pixels per projectie
%if mod(size(Radondata,1),2)==0
%    NpixPProjNew=NpixPProj+1;
%else
    NpixPProjNew = NpixPProj;
%end

%% FAN-BEAM RAYS

% flip sinogram, why?
RadonData = flipdim(RadonData,2);  %  matlab gebruikt tegengestelde draairichting (denkik) als skyscan, of er is een of andere flipdim geweest die gecorrigeerd moet worden))

% DetPixPos: distance of each detector to the ray through the origin (theta)
DetPixPos = -(NpixPProj-1)/2:(NpixPProj-1)/2;  % posities detectorpixels

% GammaStralen: alpha's? (result in radians!!)
GammaStralen = atan(DetPixPos/D); % alle met de detectorpixelposities overeenkomstige gammahoeken

% put beta (theta) and gamma (alpha) for each ray in 2D matrices
[beta gamma] = meshgrid(BetaDeg,GammaStralen);

% t: minimal distance between each ray and the ray through the origin
t = D*sin(gamma); % t-waarden overeenkomstig met de verschillende gamma's

theta = gamma*180/pi + beta;  % theta-waarden in graden overeenkomstig met verschillende gamma en beta waarden

%% PARALLEL BEAM RAYS

% DetPixPos: distance of each detector to the ray through the origin (theta)
DetPixPos = -(NpixPProjNew-1)/2:(NpixPProjNew-1)/2;  % posities detectorpixels

% GammaStralen: alpha's? (result in radians!!)
GammaStralenNew = atan(DetPixPos/D); % alle met de detectorpixelposities overeenkomstige gammahoeken

% put beta (theta) and gamma (alpha) for each ray in 2D matrices
[~, gamma] = meshgrid(BetaDeg,GammaStralenNew);

% t: minimal distance between each ray and the ray through the origin
tnew = D * sin(gamma); % t-waarden overeenkomstig met de verschillende gamma's

% calculate new t
step = (max(tnew)-min(tnew)) / (NpixPProjNew-1);
t_para = min(tnew):step:max(tnew);

[thetaNewCoord tNewCoord] = meshgrid(thetaDeg, t_para);

%% Interpolate
Interpolant = TriScatteredInterp(theta(:), t(:), RadonData(:),'nearest');
F = Interpolant(thetaNewCoord,tNewCoord);




