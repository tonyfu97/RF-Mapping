

# Plot the distribution of 
set terminal png size 1000, 500
set output 'error.png'
binwidth = 2
bin(x,width)=width*floor(x/width)
plot '/tmp/rfmap/gt_tb100_dist.txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes

