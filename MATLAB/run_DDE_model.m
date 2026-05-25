function [AF] = run_DDE_model(x,dm,tau1,tau2,TM)

%inputs
% x     time series to fit
% dm    number of points for center derivative
% tau1  delay 1
% tau2  delay 2
% TM    maximum considered delay

x = x(:)';

dx = central_deriv(x, 1:1:numel(x), dm);
dx = dx';

%normalize:
x = normalize(x);
dx = normalize(dx);

%clip data
x=x(1+dm:end-dm);
WL = length(x)-TM;

xdot = dx((1:WL)+TM); 
xtau1 = x((1:WL)+TM-tau1);
xtau2 = x((1:WL)+TM-tau2);
xtau1squared = xtau1.^2;


M = [xtau1',xtau2',xtau1squared'];
A = M\xdot(:);
fehler = sqrt(  sum(  (xdot(:)-M*A).^2  )./WL  );
AF = [A;fehler];