set term postscript eps enhanced color
set key left top
set view 45, 240, 1, 1
set size 1, 1
set xrange [0:9.001]
set yrange [9.001:0]
set xlabel "ol"
set ylabel "sl"
set pm3d
set isosamples 50, 50
    
a = 10.0;
k = 3.0;
mid = 4.0;
M = 11.0;
max(a,b)=((a>b)? a : b);
min(a,b)=((a<b)? a : b);
protectedlog10(x)=((x==0)? 0.0: log10(x));
protecteddiv(x,y)=((y==0)? 0.0: x/y);
add(a,b)=(a+b);
minus(a,b)=(a-b);
times(a,b)=(a*b);
pow(a,b)=(a**b);



set output "gp1policy.eps"

splot floor(max(min(x, add(minus(minus(max(min(x, add(minus(minus(x, y), 1.0790595), times(protectedlog10(add(minus(max(x, x), 1.0790595), max(x, x))), add(max(floor(minus(x, add(max(1.0790595, x), x))), minus(minus(max(x, x), 1.0790595), y)), minus(x, 1.0221792))))), times(0.63912493, ceil(x))), y), 1.0790595), times(protectedlog10(max(-1.0073211, x)), add(max(sin(min(ceil(cos(minus(x, 1.0790595))), min(min(x, max(minus(x, y), x)), add(min(x, max(add(y, y), x)), max(floor(minus(x, y)), ceil(max(x, min(x, x)))))))), minus(max(x, protecteddiv(max(floor(2.3035748), max(x, x)), add(y, y))), y)), minus(x, 1.0790595))))), max(protectedlog10(max(cos(min(x, max(times(0.63912493, ceil(x)), cos(protectedlog10(min(x, minus(max(x, x), 1.0790595))))))), ceil(floor(add(x, floor(times(0.63912493, ceil(x)))))))), protecteddiv(min(add(floor(x), ceil(x)), protecteddiv(ceil(minus(x, 1.0221792)), max(x, floor(times(0.63912493, floor(times(0.63912493, ceil(x)))))))), max(max(floor(2.3035748), x), cos(ceil(protecteddiv(x, 1.0919342)))))))) ti ""

set output "gp2policy.eps"

splot floor(0.5+max(sin(minus(max(min(max(minus(y, x), minus(y, x)), minus(x, 3.397377)), x), max(x, min(times(minus(x, 3.397377), x), x)))), ceil(minus(minus(x, 3.397377), max(min(y, -2.5250282), minus(y, add(max(minus(x, 3.397377), ceil(minus(minus(minus(x, 3.397377), max(protecteddiv(sin(sin(x)), minus(y, x)), minus(y, x))), max(-5.323125, minus(y, x))))), protecteddiv(add(min(max(times(minus(x, 3.397377), x), x), max(y, max(-5.323125, times(log10(floor(7.62898)), x)))), max(max(minus(x, 3.397377), minus(minus(minus(x, 3.397377), log10(x)), max(protecteddiv(times(minus(x, 3.397377), x), minus(y, x)), minus(y, x)))), sin(max(protecteddiv(floor(7.62898), minus(y, x)), protecteddiv(sin(minus(y, x)), minus(y, x)))))), ceil(x))))))))) ti ""

set output "ge1policy.eps"

splot floor(0.5+min(pow(3.49, minus(x, minus(y, min(-(protecteddiv(minus(6.45, x), 2.04)), y)))), x)) ti ""

set output "targetpolicy.eps"

splot max(min(floor(log10((a**x)/(1.0 + exp(-k*((a**(x-y)/(M-x))-mid))))), a-1), 0) ti ""
