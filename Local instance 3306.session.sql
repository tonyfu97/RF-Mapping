-- @BLOCK
DROP TABLE IF EXISTS n01_gt_top;
CREATE TABLE n01_gt_top (
    layer  VARCHAR(10) NOT NULL,
    unit   INT NOT NULL,
    mux    FLOAT,
    muy    FLOAT,
    sd1    FLOAT,
    sd2    FLOAT,
    ori    FLOAT,
    amp    FLOAT,
    offset FLOAT,
    fxvar  FLOAT
);

-- @BLOCK
LOAD DATA
INFILE ""
INTO TABLE 