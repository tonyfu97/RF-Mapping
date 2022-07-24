-- @BLOCK
CREATE TABLE n01_ (
    layer VARCHAR(10) NOT NULL,
    unit  INT NOT NULL,
    mux   FLOAT,
    muy   FLOAT,
    sd1   FLOAT,
    sd2   FLOAT,
);

-- @BLOCK
LOAD DATA
INFILE ""
INTO TABLE 