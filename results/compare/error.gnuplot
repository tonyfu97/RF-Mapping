# To run the script, do:
# gnuplot file_name

# Plot the distribution of error
# i.e., the distances between ground truth and bar mapping paradigm(s).
# conv_i = 2
set terminal png size 1000, 500
# set output 'error.png'
binwidth = 2
bin(x,width)=width*floor(x/width)
# plot "/tmp/rfmap/dist_conv".conv_i.".txt" using (bin($2,binwidth)):(1.0) smooth freq with boxes notitle


do for [conv_i=2:5] {
  infile = sprintf('/tmp/rfmap/dist_conv%d.txt', conv_i)
  outfile = sprintf('error_conv%d.png', conv_i)
  set output outfile
  set title "conv" .conv_i " error"
  plot infile using (bin($2,binwidth)):(1.0) smooth freq with boxes notitle
}