set term postscript eps enhanced color
set key right bottom
set xrange [0:9]
set yrange [-1:1]
set xlabel "Band Number"
set ylabel "Value"
set ytics -1.0,0.1
set grid

set sample 10
set pointsize 2
set output "Linear.eps"
plot\
(x-0)/10 ti "M_0" with linespoints,\
(x-1)/10 ti "M_1" with linespoints,\
(x-2)/10 ti "M_2" with linespoints,\
(x-3)/10 ti "M_3" with linespoints,\
(x-4)/10 ti "M_4" with linespoints,\
(x-5)/10 ti "M_5" with linespoints,\
(x-6)/10 ti "M_6" with linespoints,\
(x-7)/10 ti "M_7" with linespoints,\
(x-8)/10 ti "M_8" with linespoints,\
(x-9)/10 ti "M_9" with linespoints

set output "Gaussian.eps"
pdf(x,m,v) = exp(-(x-m)*(x-m)/2/v/v)/v/sqrt(2*pi)
tpdf(x,m) = (x > m) ? 1 - (5 * pdf(x,m,2)) : -(1 - 5 * pdf(x,m,2));
plot \
tpdf(x,0) ti "M_0" with linespoints,\
tpdf(x,1) ti "M_1" with linespoints,\
tpdf(x,2) ti "M_2" with linespoints,\
tpdf(x,3) ti "M_3" with linespoints,\
tpdf(x,4) ti "M_4" with linespoints,\
tpdf(x,5) ti "M_5" with linespoints,\
tpdf(x,6) ti "M_6" with linespoints,\
tpdf(x,7) ti "M_7" with linespoints,\
tpdf(x,8) ti "M_8" with linespoints,\
tpdf(x,9) ti "M_9" with linespoints
