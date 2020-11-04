import numpy as np

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
twoPi = 2. * np.pi

kittiDir = '/Volumes/hardware/KITTI/raw_data/2011_09_26'
drive = '2011_09_26_drive_0027_sync'