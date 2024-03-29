"""
Classes used to describe the columns of the result txt files. Helps with
extracting information from the txt files and put them into pandas dataframes.

Example usage
-------------
from src.rf_mapping.result_txt_format import RfmpHotSpot as HS
hot_spot_df = pd.read_csv(df_path, sep=" ", header=None)
hot_spot_df.columns = [e.name for e in HS]

Tony Fu, July 27, 2022
"""
from enum import Enum


class GtGaussian(Enum):
    LAYER  = 0  # layer name
    UNIT   = 1  # index of unit, 0...
    MUX    = 2  # mean of Gaussian along x-axis
    MUY    = 3  # mean of Gaussian along y-axis (negative)
    SIGMA1 = 4  # SIMGA on one axis
    SIGMA2 = 5  # SIMGA on orthogonal axis
    ORI    = 6  # orientation (0-180 deg) of longer axis, 0:horizontal, 45:upward to right
    AMP    = 7  # amplitude of Gaussian
    OFFSET = 8  # addivitve offset of Gaussian
    FXVAR  = 9  # fraction of explained variance


class Rfmp4aTB1(Enum):
    LAYER  = 0  # layer name
    UNIT   = 1  # index of unit, 0...
    
    TOP_I  = 2  # timulus index - highest response
    TOP_X  = 3  # x-coord
    TOP_Y  = 4  # y-coord (negative)
    TOP_R  = 5  # response value
    
    BOT_I  = 6  # timulus index - lowest response
    BOT_X  = 7  # x-coord
    BOT_Y  = 8  # y-coord (negative)
    BOT_R  = 9  # response value


class Rfmp4aTBn(Enum):
    LAYER   = 0  # layer name
    UNIT    = 1  # index of unit, 0...
    
    TOP_MUX = 2  # average x-cood for top stimuli
    TOP_MUY = 3  # average y-cood for top stimuli (negative)
    BOT_MUX = 4  # average x-cood for bottom stimuli
    BOT_MUY = 5  # average y-cood for bottom stimuli (negative)


class Rfmp4aNonOverlap(Enum):
    LAYER        = 0  # layer name
    UNIT         = 1  # index of unit, 0...
    
    TOP_X        = 2  # COM x-coord of top map
    TOP_Y        = 3  # COM y-coord (negative)
    TOP_RAD_10   = 4  # radius of 10% of mass
    TOP_RAD_50   = 5  # radius of 50% of mass
    TOP_RAD_90   = 6  # radius of 90% of mass
    TOP_NUM_BARS = 7  # the number of bars included in top map
    
    BOT_X        = 8  # COM x-coord of bottom map
    BOT_Y        = 9  # COM y-coord (negative)
    BOT_RAD_10   = 10 # radius of 10% of mass
    BOT_RAD_50   = 11 # radius of 50% of mass
    BOT_RAD_90   = 12 # radius of 90% of mass
    BOT_NUM_BARS = 13 # the number of bars included in bottom map
    

class Rfmp4aWeighted(Enum):
    LAYER  = 0  # layer name
    UNIT   = 1  # index of unit, 0...
    MUX    = 2  # mean of Gaussian along x-axis
    MUY    = 3  # mean of Gaussian along y-axis (negative)
    SIGMA1 = 4  # SIMGA on one axis
    SIGMA2 = 5  # SIMGA on orthogonal axis
    ORI    = 6  # orientation (0-180 deg) of longer axis, 0:horizontal, 45:upward to right
    AMP    = 7  # amplitude of Gaussian
    OFFSET = 8  # addivitve offset of Gaussian
    FXVAR  = 9  # fraction of explained variance
    NUM_BARS = 10  # number of bars included in the map


class RfmpCOM(Enum):
    LAYER        = 0  # layer name
    UNIT         = 1  # index of unit, 0...
    
    TOP_X        = 2  # COM x-coord of top map
    TOP_Y        = 3  # COM y-coord (negative)
    TOP_RAD_10   = 4  # radius of 10% of mass
    TOP_RAD_50   = 5  # radius of 50% of mass
    TOP_RAD_90   = 6  # radius of 90% of mass
    
    BOT_X        = 7  # COM x-coord of bottom map
    BOT_Y        = 8  # COM y-coord (negative)
    BOT_RAD_10   = 9  # radius of 10% of mass
    BOT_RAD_50   = 10 # radius of 50% of mass
    BOT_RAD_90   = 11 # radius of 90% of mass
    

class RfmpHotSpot(Enum):
    LAYER        = 0  # layer name
    UNIT         = 1  # index of unit, 0...              
                      # Top map
    TOP_X        = 2  # x-coord of max pixel
    TOP_Y        = 3  # y-coord of max pixel (negative)
                      # Bottom map
    BOT_X        = 4  # x-coord of max pixel
    BOT_Y        = 5  # y-coord of max pixel (negative)


class Rfmp4aSplist(Enum):
    STIM_I = 0  # counting from 0
    XN     = 1
    YN     = 2
    X0     = 3
    Y0     = 4
    THETA  = 5
    LEN    = 6
    WID    = 7
    AA     = 8
    FGVAL  = 9
    BGVAL  = 10


class Rfmp4c7oSplist(Enum):
    STIM_I = 0  # counting from 0
    XN     = 1
    YN     = 2
    X0     = 3
    Y0     = 4
    THETA  = 5
    LEN    = 6
    WID    = 7
    AA     = 8  # anti-alias 'thickness'
    R1     = 9  # red foreground
    G1     = 10
    B1     = 11
    R0     = 12  # red background
    G0     = 13
    B0     = 14


class PasuBWSplist(Enum):
    STIM_I = 0  # counting from 0
    XN     = 1
    YN     = 2
    X0     = 3
    Y0     = 4
    SI     = 5  # shape index [1...51]
    RI     = 6  # rotation index (could be [0...1], [0...3], or [0...7])
    FGVAL  = 7
    BGVAL  = 8
    SIZE   = 9


class CenterReponses(Enum):
    UNIT   = 0
    RANK   = 1  # top 0, 1, 2, ... and bottom 0, 1, 2, ...
    STIM_I = 2  # index of the stimulus in splist
    R      = 3  # response value (rounded to 4 decimal places)
