% Author: Yannis Papagiannakos
%
% Read data from .bin files and plot fractal

COLORMAP = [jet(); ...
            flipud( jet() ); ...
            0 0 0];

fid = fopen('../bin/count.bin','r','n');
count = fread(fid, 'double');
dims = length(count);
dims = sqrt(dims);
count = reshape(count, dims, dims).';
fclose(fid);

fid = fopen('../bin/x.bin','r','n');
x = fread(fid, 'double');
fclose(fid);

fid = fopen('../bin/y.bin','r','n');
y = fread(fid, 'double');
fclose(fid);

fig1=figure();
imagesc( x, y, count );
colormap(COLORMAP);
axis off
saveas(fig1, 'juliaset', 'epsc'); 
