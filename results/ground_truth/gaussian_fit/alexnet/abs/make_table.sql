USE rf_mapping;

-- Ground truth data based on the average gradient back-prop visualizations
-- for the top 100 image patches. A 2D Gaussian fit is performed.
--
-- Source code: borderownership/src/rf_mapping/ground_truth/gaussian_fit_script.py
DROP TABLE IF EXISTS n01_gt_top;
CREATE TABLE n01_gt_top (
    layer  VARCHAR(10) NOT NULL,		-- layer name
    unit   INT NOT NULL,				-- index of unit, 0...
    mux    FLOAT,						-- mean of Gaussian along x-axis
    muy    FLOAT,						-- mean of Gaussian along y-axis (negative)
    sd1    FLOAT,						-- SD on one axis
    sd2    FLOAT,						-- SD on orthogonal axis
    ori    FLOAT,						-- orientation (0-180 deg) of longer axis, 0:horizontal, 45:upward to right
    amp    FLOAT,						-- amplitude of Gaussian
    offset FLOAT,						-- addivitve offset of Gaussian
    fxvar  FLOAT,						-- fraction of explained variance
    PRIMARY KEY (layer, unit)
);
LOAD DATA LOCAL INFILE "/Users/tonyfu/Desktop/Bair Lab/borderownership/results/ground_truth/gaussian_fit/alexnet/abs/alexnet_gt_gaussian_top.txt"
INTO TABLE n01_gt_top
FIELDS TERMINATED BY ' '; 

-- Same as 'n01_gt_top' but for 100 images patches evoking the lowest responses.
--
-- Source code: borderownership/src/rf_mapping/ground_truth/gaussian_fit_script.py
DROP TABLE IF EXISTS n01_gt_bot;
CREATE TABLE n01_gt_bot (
    layer  VARCHAR(10) NOT NULL,		-- layer name
    unit   INT NOT NULL,				-- index of unit, 0...
    mux    FLOAT,						-- mean of Gaussian along x-axis
    muy    FLOAT,						-- mean of Gaussian along y-axis (negative)
    sd1    FLOAT,						-- SD on one axis
    sd2    FLOAT,						-- SD on orthogonal axis
    ori    FLOAT,						-- orientation (0-180 deg) of longer axis, 0:horizontal, 45:upward to right
    amp    FLOAT,						-- amplitude of Gaussian
    offset FLOAT,						-- addivitve offset of Gaussian
    fxvar  FLOAT,						-- fraction of explained variance
    PRIMARY KEY (layer, unit)
);
LOAD DATA LOCAL INFILE "/Users/tonyfu/Desktop/Bair Lab/borderownership/results/ground_truth/gaussian_fit/alexnet/abs/alexnet_gt_gaussian_bot.txt"
INTO TABLE n01_gt_bot
FIELDS TERMINATED BY ' '; 
