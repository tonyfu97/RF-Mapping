DROP TABLE IF EXISTS ALEXNET_MAX_IOU;
DROP TABLE IF EXISTS ALEXNET_MIN_IOU;
DROP TABLE IF EXISTS VGG16_MAX_IOU;
DROP TABLE IF EXISTS VGG16_MIN_IOU;
DROP TABLE IF EXISTS RESNET18_MAX_IOU;
DROP TABLE IF EXISTS RESNET18_MIN_IOU;


CREATE TABLE ALEXNET_MAX_IOU (
    LAYER INT,
    UNIT INT,
    gt_vs_gt FLOAT,
    gt_vs_gt_composite FLOAT,
    gt_vs_occlude_composite FLOAT,
    gt_vs_rfmp4a FLOAT,
    gt_vs_rfmp4c7o FLOAT,
    gt_composite_vs_gt_composite FLOAT,
    gt_composite_vs_occlude_composite FLOAT,
    gt_composite_vs_rfmp4a FLOAT,
    gt_composite_vs_rfmp4c7o FLOAT,
    occlude_composite_vs_occlude_composite FLOAT,
    occlude_composite_vs_rfmp4a FLOAT,
    occlude_composite_vs_rfmp4c7o FLOAT,
    rfmp4a_vs_rfmp4a FLOAT,
    rfmp4a_vs_rfmp4c7o FLOAT,
    rfmp4c7o_vs_rfmp4c7o FLOAT,
    PRIMARY KEY (LAYER, UNIT)
);

CREATE TABLE ALEXNET_MIN_IOU (
    LAYER INT,
    UNIT INT,
    gt_vs_gt FLOAT,
    gt_vs_gt_composite FLOAT,
    gt_vs_occlude_composite FLOAT,
    gt_vs_rfmp4a FLOAT,
    gt_vs_rfmp4c7o FLOAT,
    gt_composite_vs_gt_composite FLOAT,
    gt_composite_vs_occlude_composite FLOAT,
    gt_composite_vs_rfmp4a FLOAT,
    gt_composite_vs_rfmp4c7o FLOAT,
    occlude_composite_vs_occlude_composite FLOAT,
    occlude_composite_vs_rfmp4a FLOAT,
    occlude_composite_vs_rfmp4c7o FLOAT,
    rfmp4a_vs_rfmp4a FLOAT,
    rfmp4a_vs_rfmp4c7o FLOAT,
    rfmp4c7o_vs_rfmp4c7o FLOAT,
    PRIMARY KEY (LAYER, UNIT)
);

CREATE TABLE VGG16_MAX_IOU (
    LAYER INT,
    UNIT INT,
    gt_vs_gt FLOAT,
    gt_vs_gt_composite FLOAT,
    gt_vs_occlude_composite FLOAT,
    gt_vs_rfmp4a FLOAT,
    gt_vs_rfmp4c7o FLOAT,
    gt_composite_vs_gt_composite FLOAT,
    gt_composite_vs_occlude_composite FLOAT,
    gt_composite_vs_rfmp4a FLOAT,
    gt_composite_vs_rfmp4c7o FLOAT,
    occlude_composite_vs_occlude_composite FLOAT,
    occlude_composite_vs_rfmp4a FLOAT,
    occlude_composite_vs_rfmp4c7o FLOAT,
    rfmp4a_vs_rfmp4a FLOAT,
    rfmp4a_vs_rfmp4c7o FLOAT,
    rfmp4c7o_vs_rfmp4c7o FLOAT,
    PRIMARY KEY (LAYER, UNIT)
);

CREATE TABLE VGG16_MIN_IOU (
    LAYER INT,
    UNIT INT,
    gt_vs_gt FLOAT,
    gt_vs_gt_composite FLOAT,
    gt_vs_occlude_composite FLOAT,
    gt_vs_rfmp4a FLOAT,
    gt_vs_rfmp4c7o FLOAT,
    gt_composite_vs_gt_composite FLOAT,
    gt_composite_vs_occlude_composite FLOAT,
    gt_composite_vs_rfmp4a FLOAT,
    gt_composite_vs_rfmp4c7o FLOAT,
    occlude_composite_vs_occlude_composite FLOAT,
    occlude_composite_vs_rfmp4a FLOAT,
    occlude_composite_vs_rfmp4c7o FLOAT,
    rfmp4a_vs_rfmp4a FLOAT,
    rfmp4a_vs_rfmp4c7o FLOAT,
    rfmp4c7o_vs_rfmp4c7o FLOAT,
    PRIMARY KEY (LAYER, UNIT)
);

CREATE TABLE RESNET18_MAX_IOU (
    LAYER INT,
    UNIT INT,
    gt_vs_gt FLOAT,
    gt_vs_gt_composite FLOAT,
    gt_vs_occlude_composite FLOAT,
    gt_vs_rfmp4a FLOAT,
    gt_vs_rfmp4c7o FLOAT,
    gt_composite_vs_gt_composite FLOAT,
    gt_composite_vs_occlude_composite FLOAT,
    gt_composite_vs_rfmp4a FLOAT,
    gt_composite_vs_rfmp4c7o FLOAT,
    occlude_composite_vs_occlude_composite FLOAT,
    occlude_composite_vs_rfmp4a FLOAT,
    occlude_composite_vs_rfmp4c7o FLOAT,
    rfmp4a_vs_rfmp4a FLOAT,
    rfmp4a_vs_rfmp4c7o FLOAT,
    rfmp4c7o_vs_rfmp4c7o FLOAT,
    PRIMARY KEY (LAYER, UNIT)
);

CREATE TABLE RESNET18_MIN_IOU (
    LAYER INT,
    UNIT INT,
    gt_vs_gt FLOAT,
    gt_vs_gt_composite FLOAT,
    gt_vs_occlude_composite FLOAT,
    gt_vs_rfmp4a FLOAT,
    gt_vs_rfmp4c7o FLOAT,
    gt_composite_vs_gt_composite FLOAT,
    gt_composite_vs_occlude_composite FLOAT,
    gt_composite_vs_rfmp4a FLOAT,
    gt_composite_vs_rfmp4c7o FLOAT,
    occlude_composite_vs_occlude_composite FLOAT,
    occlude_composite_vs_rfmp4a FLOAT,
    occlude_composite_vs_rfmp4c7o FLOAT,
    rfmp4a_vs_rfmp4a FLOAT,
    rfmp4a_vs_rfmp4c7o FLOAT,
    rfmp4c7o_vs_rfmp4c7o FLOAT,
    PRIMARY KEY (LAYER, UNIT)
);

PRAGMA foreign_keys=ON;  -- to enable foreign key in sqlite
