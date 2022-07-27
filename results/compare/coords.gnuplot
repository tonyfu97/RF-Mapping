# To run the script, do:
# gnuplot file_name

# Plot the x and y coordinate of the top and bottom maps.
# Note: conv1 is skipped because almost all units fit poorly.

array xranges[6]
xranges[2] = 30
xranges[3] = 50
xranges[4] = 100
xranges[5] = 100

do for [conv_i=2:5] {
    set terminal png size 800, 2400
    infile = sprintf("/tmp/rfmap/tb_coords_conv%d.txt", conv_i)
    outfile = sprintf("conv%d_coords.png", conv_i)
    title_str = sprintf("conv%d coordinates", conv_i)
    set output outfile

    set multiplot layout 6,2 title title_str font ",20"
        set xrange[-xranges[conv_i]:xranges[conv_i]]
        set yrange[-xranges[conv_i]:xranges[conv_i]]
        set xlabel "x"
        set ylabel "y"

        set title "ground truth top"
        plot infile using 3:4 notitle
        set title "ground truth bottom"
        plot infile using 5:6 notitle
        set title "top bar"
        plot infile using 7:8 notitle
        set title "bottom bar"
        plot infile using 9:10 notitle
        set title "avg of top 20 bars"
        plot infile using 11:12 notitle
        set title "avg of bottom 20 bars"
        plot infile using 13:14 notitle
        set title "avg of top 100 bars"
        plot infile using 15:16 notitle
        set title "avg of bottom 100 bars"
        plot infile using 17:18 notitle
        set title "non overlap top"
        plot infile using 19:20 notitle
        set title "non overlap bottom"
        plot infile using 21:22 notitle
        set title "weighted top"
        plot infile using 23:24 notitle
        set title "weighted top"
        plot infile using 25:26 notitle
    unset multiplot
}