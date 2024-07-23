import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MOTION as motion
import os
dt = 4
dx = 0.02
cong_matrix, free_matrix = motion.generate_weight_matrices(delta=0.12, dx=0.02, dt=4, c_cong=12.5, c_free=-45, tau=20, plot=False)

motion_data_path = 'motion_data/'

# TODO: add a function to get the file list from you folder
entries = {}

for entry in entries:
# TODO: here only lane 1 as an example, if you would like to process more lanes, add another for look for the lane_{lane_number}_speed_matrix
    data = pd.read_csv(f'{motion_data_path}{entry}/lane_1_speed_matrix.csv')
    delta = 0.12
    tau = 20
    cong_matrix, free_matrix = motion.generate_weight_matrices(delta=delta, dx=0.02, dt=4, c_cong=12.5, c_free=-45, tau=tau, plot=False)
    smooth_speed_field = motion.smooth_speed_field(3600*data, cong_matrix, free_matrix)[:3600, :200]
    # already changed it to while loop to make sure there is no nan values in the matrix
    # main idea for this while loop is to make sure no NaN data is generated at the end of the code
    while np.isnan(smooth_speed_field).sum().sum() != 0:
        print('---- NAN values detected ----')
        delta += 0.08
        tau += 20
        cong_matrix, free_matrix = motion.generate_weight_matrices(delta=delta, dx=0.02, dt=4, c_cong=12.5, c_free=-45, tau=tau, plot=False)
        smooth_speed_field = motion.smooth_speed_field_nan(smooth_speed_field, cong_matrix, free_matrix)[:3600, :200]
        print(f'----now, there are {pd.DataFrame(smooth_speed_field).isna().sum().sum()} left----')
    # enforce it to be the exact size you want, here I enforce it to 4 hours and 4 miles
    motion_speed = smooth_speed_field[:3600, :200]
    # TODO: add the code to save the data, to the folder you'd like to
    # you may need from ASM import matrix_to_coordinates to transform a matrix to the desired coordinates with 3 columns
    # Note: the processed speed matrix after the smoothing is mph while the input is mile per second
