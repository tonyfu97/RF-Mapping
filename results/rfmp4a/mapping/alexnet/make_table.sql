USE rf_mapping;

-- Using RF mapping paradigm 4a (rfmp-4a), this reports the stimulus
-- index, coordinates, and response values for the top stimulus (maximal
-- response) and bottom stimulus (minimal response) across the set of
-- bar stimuli.
--
-- Source code: borderownership/src/rf_mapping/rfmp4a/4a_mapping_script.py
DROP TABLE IF EXISTS n01_rfmp4a_tb1;
CREATE TABLE n01_rfmp4a_tb1 (
    layer  VARCHAR(10) NOT NULL,		-- layer name
    unit   INT NOT NULL,				-- index of unit, 0...
    top_i  INT,					   		-- stimulus index - highest response
    top_x  FLOAT,						-- x-coord
    top_y  FLOAT,						-- y-coord (negative)
    top_r  FLOAT,						-- response value
    bot_i  INT,					   		-- stimulus index - lowest response
    bot_x  FLOAT,						-- x-coord
    bot_y  FLOAT,						-- y-coord (negative)
    bot_r  FLOAT,						-- response value
    PRIMARY KEY (layer, unit)
);
LOAD DATA LOCAL INFILE "/Users/tonyfu/Desktop/Bair Lab/borderownership/results/rfmp4a/mapping/alexnet/alexnet_rfmp4a_tb1.txt"
INTO TABLE n01_rfmp4a_tb1
FIELDS TERMINATED BY ' ';
-- Note: might get warning for 'duplicate entries' for conv1 because top and
--       bottom stimuli usually differ only in contrast polarities.


-- Using RF mapping paradigm 4a (rfmp-4a), this reports the mean x
-- and y coordinates of the top 20 bar stimuli and the same for the
-- bottom 20 stimuli.
--
-- Source code: borderownership/src/rf_mapping/rfmp4a/4a_mapping_script.py
DROP TABLE IF EXISTS n01_rfmp4a_tb20;
CREATE TABLE n01_rfmp4a_tb20 (
    layer     VARCHAR(10) NOT NULL,		-- layer name
    unit      INT NOT NULL,				-- index of unit, 0...
    top_mux   FLOAT,					-- average x-cood for top stimuli
    top_muy   FLOAT,					-- average y-cood for top stimuli (negative)
    bot_mux   FLOAT,					-- average x-cood for bottom stimuli
    bot_muy   FLOAT,					-- average y-cood for bottom stimuli (negative)
    PRIMARY KEY (layer, unit)
);
LOAD DATA LOCAL INFILE "/Users/tonyfu/Desktop/Bair Lab/borderownership/results/rfmp4a/mapping/alexnet/alexnet_rfmp4a_tb20.txt"
INTO TABLE n01_rfmp4a_tb20
FIELDS TERMINATED BY ' ';
-- Note: might get warning for 'duplicate entries' for conv1 because top and
--       bottom stimuli usually differ only in contrast polarities.


-- Same as 'n01_rfmp4a_tb20' but for top and bottom 100 stimuli.
DROP TABLE IF EXISTS n01_rfmp4a_tb100;
CREATE TABLE n01_rfmp4a_tb100 (
    layer     VARCHAR(10) NOT NULL,		-- layer name
    unit      INT NOT NULL,				-- index of unit, 0...
    top_mux   FLOAT,					-- average x-cood for top stimuli
    top_muy   FLOAT,					-- average y-cood for top stimuli (negative)
    bot_mux   FLOAT,					-- average x-cood for bottom stimuli
    bot_muy   FLOAT,					-- average y-cood for bottom stimuli (negative)
    PRIMARY KEY (layer, unit)
);
LOAD DATA LOCAL INFILE "/Users/tonyfu/Desktop/Bair Lab/borderownership/results/rfmp4a/mapping/alexnet/alexnet_rfmp4a_tb100.txt"
INTO TABLE n01_rfmp4a_tb100
FIELDS TERMINATED BY ' ';

