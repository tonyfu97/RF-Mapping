-- Before running the following queries:
-- 1. first create the directory '/tmp/rfmap', and make sure it
--    is owned by mysql (sudo chmod +r mysql /tmp/rfmap).
-- 2. must manually delete the files that already exist. MySQL cannot
--    delete/overwrite existing files due to security reasons.
-- 3. must have a file '/etc/my.cnf' with the content:
--         	    [mysqld]
--              secure_file_priv = ""
--    This diables the security file privileges so we can write into a
--    text file.

-- Some useful terminal commands:
-- # mannally remove existing txt files
-- cd /tmp/rfmap
-- sudo rm *
-- # after file generation, to allow gnuplot to read
-- sudo chmod +r *

USE rf_mapping;


-- The distances between the centers of the top and bottom RFs.
SELECT gt_t.unit,
	SQRT(POWER((gt_t.mux - gt_b.mux), 2)                   + POWER((gt_t.muy - gt_b.muy), 2))                   AS tb_dist,
	SQRT(POWER((bar_tb1.top_x - bar_tb1.bot_x), 2)         + POWER((bar_tb1.top_y - bar_tb1.bot_y), 2))         AS 1_dist,
	SQRT(POWER((bar_tb20.top_mux - bar_tb20.bot_mux), 2)   + POWER((bar_tb20.top_muy - bar_tb20.bot_muy), 2))   AS 20_dist,
	SQRT(POWER((bar_tb100.top_mux - bar_tb100.bot_mux), 2) + POWER((bar_tb100.top_muy - bar_tb100.bot_muy), 2)) AS 100_dist,
	SQRT(POWER((bar_no.top_x - bar_no.bot_x), 2)           + POWER((bar_no.top_y - bar_no.bot_y), 2))           AS no_dist,
	SQRT(POWER((bar_w_t.mux - bar_w_b.mux), 2)             + POWER((bar_w_t.muy - bar_w_b.muy), 2))             AS w_dist
FROM n01_gt_top AS gt_t,
	n01_gt_bot AS gt_b,
	n01_rfmp4a_tb1 AS bar_tb1,
	n01_rfmp4a_tb20 AS bar_tb20,
	n01_rfmp4a_tb100 AS bar_tb100,
	n01_rfmp4a_non_overlap AS bar_no,
	n01_rfmp4a_weighted_top AS bar_w_t,
	n01_rfmp4a_weighted_bot AS bar_w_b
WHERE gt_t.layer = 'conv2' AND gt_t.fxvar > 0.8 AND gt_b.fxvar > 0.8 AND
	bar_no.top_rad_10 != -1 AND bar_no.top_rad_10 != -1 AND
	bar_w_t.fxvar > 0.8 AND bar_w_b.fxvar > 0.8 AND
	gt_t.layer = gt_b.layer AND gt_t.unit = gt_b.unit AND
	gt_b.layer = bar_tb1.layer AND gt_b.unit = bar_tb1.unit AND
	bar_tb1.layer = bar_tb20.layer AND bar_tb1.unit = bar_tb20.unit AND
	bar_tb20.layer = bar_tb100.layer AND bar_tb20.unit = bar_tb100.unit AND
	bar_tb100.layer = bar_no.layer AND bar_tb100.unit = bar_no.unit AND
	bar_no.layer = bar_w_t.layer AND bar_no.unit = bar_w_t.unit AND
	bar_w_t.layer = bar_w_b.layer AND bar_w_t.unit = bar_w_b.unit
INTO OUTFILE "/tmp/rfmap/dist_conv2.txt";


SELECT gt_t.unit,
	SQRT(POWER((gt_t.mux - gt_b.mux), 2)                   + POWER((gt_t.muy - gt_b.muy), 2))                   AS tb_dist,
	SQRT(POWER((bar_tb1.top_x - bar_tb1.bot_x), 2)         + POWER((bar_tb1.top_y - bar_tb1.bot_y), 2))         AS 1_dist,
	SQRT(POWER((bar_tb20.top_mux - bar_tb20.bot_mux), 2)   + POWER((bar_tb20.top_muy - bar_tb20.bot_muy), 2))   AS 20_dist,
	SQRT(POWER((bar_tb100.top_mux - bar_tb100.bot_mux), 2) + POWER((bar_tb100.top_muy - bar_tb100.bot_muy), 2)) AS 100_dist,
	SQRT(POWER((bar_no.top_x - bar_no.bot_x), 2)           + POWER((bar_no.top_y - bar_no.bot_y), 2))           AS no_dist,
	SQRT(POWER((bar_w_t.mux - bar_w_b.mux), 2)             + POWER((bar_w_t.muy - bar_w_b.muy), 2))             AS w_dist
FROM n01_gt_top AS gt_t,
	n01_gt_bot AS gt_b,
	n01_rfmp4a_tb1 AS bar_tb1,
	n01_rfmp4a_tb20 AS bar_tb20,
	n01_rfmp4a_tb100 AS bar_tb100,
	n01_rfmp4a_non_overlap AS bar_no,
	n01_rfmp4a_weighted_top AS bar_w_t,
	n01_rfmp4a_weighted_bot AS bar_w_b
WHERE gt_t.layer = 'conv3' AND gt_t.fxvar > 0.8 AND gt_b.fxvar > 0.8 AND
	bar_no.top_rad_10 != -1 AND bar_no.top_rad_10 != -1 AND
	bar_w_t.fxvar > 0.8 AND bar_w_b.fxvar > 0.8 AND
	gt_t.layer = gt_b.layer AND gt_t.unit = gt_b.unit AND
	gt_b.layer = bar_tb1.layer AND gt_b.unit = bar_tb1.unit AND
	bar_tb1.layer = bar_tb20.layer AND bar_tb1.unit = bar_tb20.unit AND
	bar_tb20.layer = bar_tb100.layer AND bar_tb20.unit = bar_tb100.unit AND
	bar_tb100.layer = bar_no.layer AND bar_tb100.unit = bar_no.unit AND
	bar_no.layer = bar_w_t.layer AND bar_no.unit = bar_w_t.unit AND
	bar_w_t.layer = bar_w_b.layer AND bar_w_t.unit = bar_w_b.unit
INTO OUTFILE "/tmp/rfmap/dist_conv3.txt";


SELECT gt_t.unit,
	SQRT(POWER((gt_t.mux - gt_b.mux), 2)                   + POWER((gt_t.muy - gt_b.muy), 2))                   AS tb_dist,
	SQRT(POWER((bar_tb1.top_x - bar_tb1.bot_x), 2)         + POWER((bar_tb1.top_y - bar_tb1.bot_y), 2))         AS 1_dist,
	SQRT(POWER((bar_tb20.top_mux - bar_tb20.bot_mux), 2)   + POWER((bar_tb20.top_muy - bar_tb20.bot_muy), 2))   AS 20_dist,
	SQRT(POWER((bar_tb100.top_mux - bar_tb100.bot_mux), 2) + POWER((bar_tb100.top_muy - bar_tb100.bot_muy), 2)) AS 100_dist,
	SQRT(POWER((bar_no.top_x - bar_no.bot_x), 2)           + POWER((bar_no.top_y - bar_no.bot_y), 2))           AS no_dist,
	SQRT(POWER((bar_w_t.mux - bar_w_b.mux), 2)             + POWER((bar_w_t.muy - bar_w_b.muy), 2))             AS w_dist
FROM n01_gt_top AS gt_t,
	n01_gt_bot AS gt_b,
	n01_rfmp4a_tb1 AS bar_tb1,
	n01_rfmp4a_tb20 AS bar_tb20,
	n01_rfmp4a_tb100 AS bar_tb100,
	n01_rfmp4a_non_overlap AS bar_no,
	n01_rfmp4a_weighted_top AS bar_w_t,
	n01_rfmp4a_weighted_bot AS bar_w_b
WHERE gt_t.layer = 'conv4' AND gt_t.fxvar > 0.8 AND gt_b.fxvar > 0.8 AND
	bar_no.top_rad_10 != -1 AND bar_no.top_rad_10 != -1 AND
	bar_w_t.fxvar > 0.8 AND bar_w_b.fxvar > 0.8 AND
	gt_t.layer = gt_b.layer AND gt_t.unit = gt_b.unit AND
	gt_b.layer = bar_tb1.layer AND gt_b.unit = bar_tb1.unit AND
	bar_tb1.layer = bar_tb20.layer AND bar_tb1.unit = bar_tb20.unit AND
	bar_tb20.layer = bar_tb100.layer AND bar_tb20.unit = bar_tb100.unit AND
	bar_tb100.layer = bar_no.layer AND bar_tb100.unit = bar_no.unit AND
	bar_no.layer = bar_w_t.layer AND bar_no.unit = bar_w_t.unit AND
	bar_w_t.layer = bar_w_b.layer AND bar_w_t.unit = bar_w_b.unit
INTO OUTFILE "/tmp/rfmap/dist_conv4.txt";


SELECT gt_t.unit,
	SQRT(POWER((gt_t.mux - gt_b.mux), 2)                   + POWER((gt_t.muy - gt_b.muy), 2))                   AS tb_dist,
	SQRT(POWER((bar_tb1.top_x - bar_tb1.bot_x), 2)         + POWER((bar_tb1.top_y - bar_tb1.bot_y), 2))         AS 1_dist,
	SQRT(POWER((bar_tb20.top_mux - bar_tb20.bot_mux), 2)   + POWER((bar_tb20.top_muy - bar_tb20.bot_muy), 2))   AS 20_dist,
	SQRT(POWER((bar_tb100.top_mux - bar_tb100.bot_mux), 2) + POWER((bar_tb100.top_muy - bar_tb100.bot_muy), 2)) AS 100_dist,
	SQRT(POWER((bar_no.top_x - bar_no.bot_x), 2)           + POWER((bar_no.top_y - bar_no.bot_y), 2))           AS no_dist,
	SQRT(POWER((bar_w_t.mux - bar_w_b.mux), 2)             + POWER((bar_w_t.muy - bar_w_b.muy), 2))             AS w_dist
FROM n01_gt_top AS gt_t,
	n01_gt_bot AS gt_b,
	n01_rfmp4a_tb1 AS bar_tb1,
	n01_rfmp4a_tb20 AS bar_tb20,
	n01_rfmp4a_tb100 AS bar_tb100,
	n01_rfmp4a_non_overlap AS bar_no,
	n01_rfmp4a_weighted_top AS bar_w_t,
	n01_rfmp4a_weighted_bot AS bar_w_b
WHERE gt_t.layer = 'conv5' AND gt_t.fxvar > 0.8 AND gt_b.fxvar > 0.8 AND
	bar_no.top_rad_10 != -1 AND bar_no.top_rad_10 != -1 AND
	bar_w_t.fxvar > 0.8 AND bar_w_b.fxvar > 0.8 AND
	gt_t.layer = gt_b.layer AND gt_t.unit = gt_b.unit AND
	gt_b.layer = bar_tb1.layer AND gt_b.unit = bar_tb1.unit AND
	bar_tb1.layer = bar_tb20.layer AND bar_tb1.unit = bar_tb20.unit AND
	bar_tb20.layer = bar_tb100.layer AND bar_tb20.unit = bar_tb100.unit AND
	bar_tb100.layer = bar_no.layer AND bar_tb100.unit = bar_no.unit AND
	bar_no.layer = bar_w_t.layer AND bar_no.unit = bar_w_t.unit AND
	bar_w_t.layer = bar_w_b.layer AND bar_w_t.unit = bar_w_b.unit
INTO OUTFILE "/tmp/rfmap/dist_conv5.txt";



-- The error (Euclidean distance) between the RF centers of the ground truth and the bar mapping results.
SELECT gt_t.unit,
	   SQRT(POWER((gt_t.mux - gt_b.mux), 2)          + POWER((gt_t.muy - gt_b.muy), 2))          AS tb_dist,
	   SQRT(POWER((gt_t.mux - bar_tb1.top_x), 2)     + POWER((gt_t.muy - bar_tb1.top_y), 2))     AS t1_err,
	   SQRT(POWER((gt_t.mux - bar_tb1.bot_x), 2)     + POWER((gt_t.muy - bar_tb1.bot_y), 2))     AS b1_err,
	   SQRT(POWER((gt_t.mux - bar_tb20.top_mux), 2)  + POWER((gt_t.muy - bar_tb20.top_muy), 2))  AS t20_err,
	   SQRT(POWER((gt_t.mux - bar_tb20.bot_mux), 2)  + POWER((gt_t.muy - bar_tb20.bot_muy), 2))  AS b20_err,
	   SQRT(POWER((gt_t.mux - bar_tb100.top_mux), 2) + POWER((gt_t.muy - bar_tb100.top_muy), 2)) AS t100_err,
	   SQRT(POWER((gt_t.mux - bar_tb100.bot_mux), 2) + POWER((gt_t.muy - bar_tb100.bot_muy), 2)) AS b100_err,
	   SQRT(POWER((gt_t.mux - bar_no.top_x), 2)      + POWER((gt_t.muy - bar_no.top_y), 2))      AS no_t_err,
	   SQRT(POWER((gt_t.mux - bar_no.bot_x), 2)      + POWER((gt_t.muy - bar_no.bot_y), 2))      AS no_b_err,
	   SQRT(POWER((gt_t.mux - bar_w_t.mux), 2)       + POWER((gt_t.muy - bar_w_t.muy), 2))       AS w_t_err,
	   SQRT(POWER((gt_t.mux - bar_w_b.mux), 2)       + POWER((gt_t.muy - bar_w_b.muy), 2))       AS w_b_err
FROM n01_gt_top AS gt_t,
	 n01_gt_bot AS gt_b,
	 n01_rfmp4a_tb1 AS bar_tb1,
	 n01_rfmp4a_tb20 AS bar_tb20,
	 n01_rfmp4a_tb100 AS bar_tb100,
	 n01_rfmp4a_non_overlap AS bar_no,
	 n01_rfmp4a_weighted_top AS bar_w_t,
	 n01_rfmp4a_weighted_bot AS bar_w_b
WHERE gt_t.layer = 'conv2' AND gt_t.fxvar > 0.8 AND gt_b.fxvar > 0.8 AND
      bar_no.top_rad_10 != -1 AND bar_no.top_rad_10 != -1 AND
	  bar_w_t.fxvar > 0.8 AND bar_w_b.fxvar > 0.8 AND
	  gt_t.layer = gt_b.layer AND gt_t.unit = gt_b.unit AND
	  gt_b.layer = bar_tb1.layer AND gt_b.unit = bar_tb1.unit AND
	  bar_tb1.layer = bar_tb20.layer AND bar_tb1.unit = bar_tb20.unit AND
	  bar_tb20.layer = bar_tb100.layer AND bar_tb20.unit = bar_tb100.unit AND
	  bar_tb100.layer = bar_no.layer AND bar_tb100.unit = bar_no.unit AND
	  bar_no.layer = bar_w_t.layer AND bar_no.unit = bar_w_t.unit AND
	  bar_w_t.layer = bar_w_b.layer AND bar_w_t.unit = bar_w_b.unit
INTO OUTFILE "/tmp/rfmap/barmap_err_conv2.txt";


SELECT gt_t.unit,
	   SQRT(POWER((gt_t.mux - gt_b.mux), 2)          + POWER((gt_t.muy - gt_b.muy), 2))          AS tb_dist,
	   SQRT(POWER((gt_t.mux - bar_tb1.top_x), 2)     + POWER((gt_t.muy - bar_tb1.top_y), 2))     AS t1_err,
	   SQRT(POWER((gt_t.mux - bar_tb1.bot_x), 2)     + POWER((gt_t.muy - bar_tb1.bot_y), 2))     AS b1_err,
	   SQRT(POWER((gt_t.mux - bar_tb20.top_mux), 2)  + POWER((gt_t.muy - bar_tb20.top_muy), 2))  AS t20_err,
	   SQRT(POWER((gt_t.mux - bar_tb20.bot_mux), 2)  + POWER((gt_t.muy - bar_tb20.bot_muy), 2))  AS b20_err,
	   SQRT(POWER((gt_t.mux - bar_tb100.top_mux), 2) + POWER((gt_t.muy - bar_tb100.top_muy), 2)) AS t100_err,
	   SQRT(POWER((gt_t.mux - bar_tb100.bot_mux), 2) + POWER((gt_t.muy - bar_tb100.bot_muy), 2)) AS b100_err,
	   SQRT(POWER((gt_t.mux - bar_no.top_x), 2)      + POWER((gt_t.muy - bar_no.top_y), 2))      AS no_t_err,
	   SQRT(POWER((gt_t.mux - bar_no.bot_x), 2)      + POWER((gt_t.muy - bar_no.bot_y), 2))      AS no_b_err,
	   SQRT(POWER((gt_t.mux - bar_w_t.mux), 2)       + POWER((gt_t.muy - bar_w_t.muy), 2))       AS w_t_err,
	   SQRT(POWER((gt_t.mux - bar_w_b.mux), 2)       + POWER((gt_t.muy - bar_w_b.muy), 2))       AS w_b_err
FROM n01_gt_top AS gt_t,
	 n01_gt_bot AS gt_b,
	 n01_rfmp4a_tb1 AS bar_tb1,
	 n01_rfmp4a_tb20 AS bar_tb20,
	 n01_rfmp4a_tb100 AS bar_tb100,
	 n01_rfmp4a_non_overlap AS bar_no,
	 n01_rfmp4a_weighted_top AS bar_w_t,
	 n01_rfmp4a_weighted_bot AS bar_w_b
WHERE gt_t.layer = 'conv3' AND gt_t.fxvar > 0.8 AND gt_b.fxvar > 0.8 AND
      bar_no.top_rad_10 != -1 AND bar_no.top_rad_10 != -1 AND
	  bar_w_t.fxvar > 0.8 AND bar_w_b.fxvar > 0.8 AND
	  gt_t.layer = gt_b.layer AND gt_t.unit = gt_b.unit AND
	  gt_b.layer = bar_tb1.layer AND gt_b.unit = bar_tb1.unit AND
	  bar_tb1.layer = bar_tb20.layer AND bar_tb1.unit = bar_tb20.unit AND
	  bar_tb20.layer = bar_tb100.layer AND bar_tb20.unit = bar_tb100.unit AND
	  bar_tb100.layer = bar_no.layer AND bar_tb100.unit = bar_no.unit AND
	  bar_no.layer = bar_w_t.layer AND bar_no.unit = bar_w_t.unit AND
	  bar_w_t.layer = bar_w_b.layer AND bar_w_t.unit = bar_w_b.unit
INTO OUTFILE "/tmp/rfmap/barmap_err_conv3.txt";


SELECT gt_t.unit,
	   SQRT(POWER((gt_t.mux - gt_b.mux), 2)          + POWER((gt_t.muy - gt_b.muy), 2))          AS tb_dist,
	   SQRT(POWER((gt_t.mux - bar_tb1.top_x), 2)     + POWER((gt_t.muy - bar_tb1.top_y), 2))     AS t1_err,
	   SQRT(POWER((gt_t.mux - bar_tb1.bot_x), 2)     + POWER((gt_t.muy - bar_tb1.bot_y), 2))     AS b1_err,
	   SQRT(POWER((gt_t.mux - bar_tb20.top_mux), 2)  + POWER((gt_t.muy - bar_tb20.top_muy), 2))  AS t20_err,
	   SQRT(POWER((gt_t.mux - bar_tb20.bot_mux), 2)  + POWER((gt_t.muy - bar_tb20.bot_muy), 2))  AS b20_err,
	   SQRT(POWER((gt_t.mux - bar_tb100.top_mux), 2) + POWER((gt_t.muy - bar_tb100.top_muy), 2)) AS t100_err,
	   SQRT(POWER((gt_t.mux - bar_tb100.bot_mux), 2) + POWER((gt_t.muy - bar_tb100.bot_muy), 2)) AS b100_err,
	   SQRT(POWER((gt_t.mux - bar_no.top_x), 2)      + POWER((gt_t.muy - bar_no.top_y), 2))      AS no_t_err,
	   SQRT(POWER((gt_t.mux - bar_no.bot_x), 2)      + POWER((gt_t.muy - bar_no.bot_y), 2))      AS no_b_err,
	   SQRT(POWER((gt_t.mux - bar_w_t.mux), 2)       + POWER((gt_t.muy - bar_w_t.muy), 2))       AS w_t_err,
	   SQRT(POWER((gt_t.mux - bar_w_b.mux), 2)       + POWER((gt_t.muy - bar_w_b.muy), 2))       AS w_b_err
FROM n01_gt_top AS gt_t,
	 n01_gt_bot AS gt_b,
	 n01_rfmp4a_tb1 AS bar_tb1,
	 n01_rfmp4a_tb20 AS bar_tb20,
	 n01_rfmp4a_tb100 AS bar_tb100,
	 n01_rfmp4a_non_overlap AS bar_no,
	 n01_rfmp4a_weighted_top AS bar_w_t,
	 n01_rfmp4a_weighted_bot AS bar_w_b
WHERE gt_t.layer = 'conv4' AND gt_t.fxvar > 0.8 AND gt_b.fxvar > 0.8 AND
      bar_no.top_rad_10 != -1 AND bar_no.top_rad_10 != -1 AND
	  bar_w_t.fxvar > 0.8 AND bar_w_b.fxvar > 0.8 AND
	  gt_t.layer = gt_b.layer AND gt_t.unit = gt_b.unit AND
	  gt_b.layer = bar_tb1.layer AND gt_b.unit = bar_tb1.unit AND
	  bar_tb1.layer = bar_tb20.layer AND bar_tb1.unit = bar_tb20.unit AND
	  bar_tb20.layer = bar_tb100.layer AND bar_tb20.unit = bar_tb100.unit AND
	  bar_tb100.layer = bar_no.layer AND bar_tb100.unit = bar_no.unit AND
	  bar_no.layer = bar_w_t.layer AND bar_no.unit = bar_w_t.unit AND
	  bar_w_t.layer = bar_w_b.layer AND bar_w_t.unit = bar_w_b.unit
INTO OUTFILE "/tmp/rfmap/barmap_err_conv4.txt";


SELECT gt_t.unit,
	   SQRT(POWER((gt_t.mux - gt_b.mux), 2)          + POWER((gt_t.muy - gt_b.muy), 2))          AS tb_dist,
	   SQRT(POWER((gt_t.mux - bar_tb1.top_x), 2)     + POWER((gt_t.muy - bar_tb1.top_y), 2))     AS t1_err,
	   SQRT(POWER((gt_t.mux - bar_tb1.bot_x), 2)     + POWER((gt_t.muy - bar_tb1.bot_y), 2))     AS b1_err,
	   SQRT(POWER((gt_t.mux - bar_tb20.top_mux), 2)  + POWER((gt_t.muy - bar_tb20.top_muy), 2))  AS t20_err,
	   SQRT(POWER((gt_t.mux - bar_tb20.bot_mux), 2)  + POWER((gt_t.muy - bar_tb20.bot_muy), 2))  AS b20_err,
	   SQRT(POWER((gt_t.mux - bar_tb100.top_mux), 2) + POWER((gt_t.muy - bar_tb100.top_muy), 2)) AS t100_err,
	   SQRT(POWER((gt_t.mux - bar_tb100.bot_mux), 2) + POWER((gt_t.muy - bar_tb100.bot_muy), 2)) AS b100_err,
	   SQRT(POWER((gt_t.mux - bar_no.top_x), 2)      + POWER((gt_t.muy - bar_no.top_y), 2))      AS no_t_err,
	   SQRT(POWER((gt_t.mux - bar_no.bot_x), 2)      + POWER((gt_t.muy - bar_no.bot_y), 2))      AS no_b_err,
	   SQRT(POWER((gt_t.mux - bar_w_t.mux), 2)       + POWER((gt_t.muy - bar_w_t.muy), 2))       AS w_t_err,
	   SQRT(POWER((gt_t.mux - bar_w_b.mux), 2)       + POWER((gt_t.muy - bar_w_b.muy), 2))       AS w_b_err
FROM n01_gt_top AS gt_t,
	 n01_gt_bot AS gt_b,
	 n01_rfmp4a_tb1 AS bar_tb1,
	 n01_rfmp4a_tb20 AS bar_tb20,
	 n01_rfmp4a_tb100 AS bar_tb100,
	 n01_rfmp4a_non_overlap AS bar_no,
	 n01_rfmp4a_weighted_top AS bar_w_t,
	 n01_rfmp4a_weighted_bot AS bar_w_b
WHERE gt_t.layer = 'conv5' AND gt_t.fxvar > 0.8 AND gt_b.fxvar > 0.8 AND
      bar_no.top_rad_10 != -1 AND bar_no.top_rad_10 != -1 AND
	  bar_w_t.fxvar > 0.8 AND bar_w_b.fxvar > 0.8 AND
	  gt_t.layer = gt_b.layer AND gt_t.unit = gt_b.unit AND
	  gt_b.layer = bar_tb1.layer AND gt_b.unit = bar_tb1.unit AND
	  bar_tb1.layer = bar_tb20.layer AND bar_tb1.unit = bar_tb20.unit AND
	  bar_tb20.layer = bar_tb100.layer AND bar_tb20.unit = bar_tb100.unit AND
	  bar_tb100.layer = bar_no.layer AND bar_tb100.unit = bar_no.unit AND
	  bar_no.layer = bar_w_t.layer AND bar_no.unit = bar_w_t.unit AND
	  bar_w_t.layer = bar_w_b.layer AND bar_w_t.unit = bar_w_b.unit
INTO OUTFILE "/tmp/rfmap/barmap_err_conv5.txt";
