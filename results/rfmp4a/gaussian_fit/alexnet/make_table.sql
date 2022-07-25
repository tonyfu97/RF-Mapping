USE rf_mapping;

-- Using RF mapping paradigm 4a (rfmp-4a), this reports the coordinates and
-- radii of the 'non-overlap bar maps', which are constructed by sequentially
-- adding bars (ordered by the responses) that result in at least of 10% of
-- maximum/minimum response. A new bar (always white on black) is added only
-- if it does not overlap with any existing bars. Note: The threshold response
-- of the top map cannot be negative, and that of the bottom map cannot be
-- positive. Therefore, some top maps may be blank because maximum response
-- is negative.
--
-- Source code: borderownership/src/rf_mapping/rfmp4a/4a_com_non_overlap_script.py
DROP TABLE IF EXISTS n01_rfmp4a_non_overlap;
CREATE TABLE n01_rfmp4a_non_overlap (
    layer         VARCHAR(10) NOT NULL,			-- layer name
    unit          INT NOT NULL,					-- index of unit, 0...
    top_x  	      FLOAT,						-- COM x-coord of top map
    top_y         FLOAT,						-- COM y-coord (negative)
	top_rad_10    FLOAT,						-- radius of 10% of mass
    top_rad_50    FLOAT,						-- radius of 50% of mass
    top_rad_90    FLOAT,						-- radius of 90% of mass
    top_num_bars  INT,							-- the number of bars included in top map
    bot_x  	      FLOAT,						-- COM x-coord of bottom map
    bot_y         FLOAT,						-- COM y-coord (negative)
	bot_rad_10    FLOAT,						-- radius of 10% of mass
    bot_rad_50    FLOAT,						-- radius of 50% of mass
    bot_rad_90    FLOAT,						-- radius of 90% of mass
    bot_num_bars  INT,							-- the number of bars included in bottom map
    PRIMARY KEY (layer, unit)
);
LOAD DATA LOCAL INFILE "/Users/tonyfu/Desktop/Bair Lab/borderownership/results/rfmp4a/gaussian_fit/alexnet/non_overlap.txt"
INTO TABLE n01_rfmp4a_non_overlap
FIELDS TERMINATED BY ' ';
-- NOTE: A COM or radius value of -1 means there is no center of mass (COM).


-- Using RF mapping paradigm 4a (rfmp-4a), this reports the coordinates and
-- radii of the 'weighted bar maps', which are constructed by sequentially
-- adding bars (ordered by the responses) that result in at least of 10% of
-- maximum/minimum response. A new bar (always white on black) is multipled
-- by the response and added to the map. Note: The threshold response
-- of the top map cannot be negative, and that of the bottom map cannot be
-- positive. Therefore, some top maps may be blank because maximum response
-- is negative.
--
-- Source code: borderownership/src/rf_mapping/rfmp4a/4a_gaussian_weighted_script.py
DROP TABLE IF EXISTS n01_rfmp4a_weighted_top;
CREATE TABLE n01_rfmp4a_weighted_top (
    layer     VARCHAR(10) NOT NULL,			-- layer name
    unit      INT NOT NULL,					-- index of unit, 0...
    mux       FLOAT,						-- mean of Gaussian along x-axis
    muy       FLOAT,						-- mean of Gaussian along y-axis (negative)
    sd1       FLOAT,						-- SD on one axis
    sd2       FLOAT,						-- SD on orthogonal axis
    ori    	  FLOAT,						-- orientation (0-180 deg) of longer axis, 0:horizontal, 45:upward to right
    amp       FLOAT,						-- amplitude of Gaussian
    offset    FLOAT,						-- addivitve offset of Gaussian
    fxvar     FLOAT,						-- fraction of explained variance
    num_bars  INT,							-- number of bars included in the map
    PRIMARY KEY (layer, unit)
);
LOAD DATA LOCAL INFILE "/Users/tonyfu/Desktop/Bair Lab/borderownership/results/rfmp4a/gaussian_fit/alexnet/weighted_top.txt"
INTO TABLE n01_rfmp4a_weighted_top
FIELDS TERMINATED BY ' '; 
-- Note: Might get 'Data truncated for column fxvar' warning because those
--       maps are empty (threshold response is 0).


-- Same as 'n01_rfmp4a_weighted_top' but for the bottom stimuli.
DROP TABLE IF EXISTS n01_rfmp4a_weighted_bot;
CREATE TABLE n01_rfmp4a_weighted_bot (
    layer     VARCHAR(10) NOT NULL,			-- layer name
    unit      INT NOT NULL,					-- index of unit, 0...
    mux       FLOAT,						-- mean of Gaussian along x-axis
    muy       FLOAT,						-- mean of Gaussian along y-axis (negative)
    sd1       FLOAT,						-- SD on one axis
    sd2       FLOAT,						-- SD on orthogonal axis
    ori    	  FLOAT,						-- orientation (0-180 deg) of longer axis, 0:horizontal, 45:upward to right
    amp       FLOAT,						-- amplitude of Gaussian
    offset    FLOAT,						-- addivitve offset of Gaussian
    fxvar     FLOAT,						-- fraction of explained variance
    num_bars  INT,							-- number of bars included in the map
    PRIMARY KEY (layer, unit)
);
LOAD DATA LOCAL INFILE "/Users/tonyfu/Desktop/Bair Lab/borderownership/results/rfmp4a/gaussian_fit/alexnet/weighted_bot.txt"
INTO TABLE n01_rfmp4a_weighted_bot
FIELDS TERMINATED BY ' '; 
