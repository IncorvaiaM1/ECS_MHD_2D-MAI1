%{
Inspect the eigenvectors of J+I for continuation.
%}

clear;
load("../null_vectors/R.mat");

clf;
tiledlayout(1,2);

nexttile;
imagesc(R);
axis square;
colorbar();
clim([-1 1]);

nexttile
e = eigs(R, size(R,1));
scatter( real(e), imag(e), 'filled' );
hold on
  th = linspace(0,7,128);
  plot(sin(th), cos(th));
hold off

f = @(x) x(2) - x(1);

%realistic aspect ratio so the unit circle is not distorted
pbaspect( [ f(xlim), f(ylim), 1] )

%% Inspect a vector

filename = "../null_vectors/null_vec.mat";
load(filename);

tl = tiledlayout(1,2)

sx
T

nexttile
imagesc( squeeze(fields(1,:,:)).' ); 
axis square; clim([-1,1] * max(abs(fields), [], "all"));
colorbar();
title("vorticity perturbation");

nexttile
imagesc( squeeze(fields(2,:,:)).' );
axis square; clim([-1,1] * max(abs(fields), [], "all"));
colorbar();
title("current perturbation");

