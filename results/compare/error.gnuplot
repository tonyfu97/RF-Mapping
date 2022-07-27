# To run the script, do:
# gnuplot file_name

# Plot the distribution of error
# i.e., the distances between ground truth and bar mapping paradigm(s).
# conv_i = 2
set terminal postscript # size 1000, 500
# set output 'error.png'
binwidth = 0.25
bin(x,width)=width*floor(x/width)
# plot "/tmp/rfmap/dist_conv".conv_i.".txt" using (bin($2,binwidth)):(1.0) smooth freq with boxes notitle


do for [conv_i=2:5] {
  infile = sprintf('/tmp/rfmap/dist_conv%d.txt', conv_i)
  outfile = sprintf('error_conv%d.eps', conv_i)
  title_str = sprintf('conv%d error', conv_i)
  set autoscale
  set output outfile
  set title title_str
  plot infile using (bin($2,binwidth)):(1.0) smooth freq with boxes notitle
}

