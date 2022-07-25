-- Before running the following queries:
-- 1. first create the directory '/tmp/rfmap', and make sure it
--    is owned by mysql (sudo chmod mysql /tmp/rfmap).
-- 2. must manually delete the files that already exist. MySQL cannot
--    delete/overwrite existing files due to security reasons.
--

USE rf_mapping;


SELECT SQRT(POWER((t.mux - b.mux), 2) + POWER((t.muy - b.muy), 2))
FROM n01_gt_top AS t, n01_gt_bot AS b
WHERE t.layer = b.layer AND
	  t.unit = b.unit
INTO OUTFILE "/tmp/rfmap/gt_tb_dist.txt";


SELECT SQRT(POWER((gt.mux - bar.top_x), 2) + POWER((gt.muy - bar.top_y), 2))
FROM n01_gt_top AS gt, n01_rfmp4a_non_overlap as bar
WHERE gt.layer = bar.layer AND
	  gt.unit = bar.unit
INTO OUTFILE "/tmp/rfmap/gt_non_overlap_bar_dist.txt";


SELECT SQRT(POWER((gt.mux - bar.top_x), 2) + POWER((gt.muy - bar.top_y), 2))
FROM n01_gt_top AS gt, n01_rfmp4a_tb1 as bar
WHERE gt.layer = bar.layer AND
	  gt.unit = bar.unit
INTO OUTFILE "/tmp/rfmap/gt_tb1_dist.txt";

SELECT SQRT(POWER((gt.mux - bar.top_mux), 2) + POWER((gt.muy - bar.top_muy), 2))
FROM n01_gt_top AS gt, n01_rfmp4a_tb20 as bar
WHERE gt.layer = bar.layer AND
	  gt.unit = bar.unit
INTO OUTFILE "/tmp/rfmap/gt_tb20_dist.txt";

SELECT SQRT(POWER((gt.mux - bar.top_mux), 2) + POWER((gt.muy - bar.top_muy), 2))
FROM n01_gt_top AS gt, n01_rfmp4a_tb100 as bar
WHERE gt.layer = bar.layer AND
	  gt.unit = bar.unit
INTO OUTFILE "/tmp/rfmap/gt_tb100_dist.txt";