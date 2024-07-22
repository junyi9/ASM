import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def add_bounded_edges(matrix, boundary_value, row_boundary_thickness, col_boundary_thickness):
    original_rows, original_cols = matrix.shape
    new_rows = original_rows + 2 * row_boundary_thickness
    new_cols = original_cols + 2 * col_boundary_thickness

    # Create a new matrix filled with the boundary value
    new_matrix = np.full((new_rows, new_cols), boundary_value)

    # Insert the original matrix into the center of the new matrix
    new_matrix[row_boundary_thickness:row_boundary_thickness + original_rows,
               col_boundary_thickness:col_boundary_thickness + original_cols] = matrix

    return new_matrix

def generate_weight_matrices(delta=0.12, dx=0.02, dt=4, c_cong=13, c_free=-45, tau=20, plot=True):
    t = abs(delta / c_cong / 2)
    x_mat = 2*int(delta/dx/2) + 1
    t_mat = int(t / dt * 3600) * 2 + 1
    matrix = np.zeros([x_mat, t_mat])
    matrix_df = pd.DataFrame(matrix)
    st_df = matrix_df.stack().reset_index()
    st_df.columns = ['x', 't', 'weight']
    st_df['time'] = dt * (st_df['t'] - int(t_mat / 2))
    st_df['space'] = dx * (st_df['x'] - int(x_mat / 2))

    def fill_cong_weight(row):
        t_new = row['time'] - row['space'] / (c_cong / 3600)
        if abs(t_new) < tau / 2:
            return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))
        else:
            return 0

    def fill_free_weight(row):
        t_new = row['time'] - row['space'] / (c_free / 3600)
        if abs(t_new) < tau / 2:
            return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))
        else:
            return 0

    st_df['cong_weight'] = st_df.apply(fill_cong_weight, axis=1)
    st_df['free_weight'] = st_df.apply(fill_free_weight, axis=1)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=20)
        plt.scatter(st_df.time, -st_df.space, c=st_df.cong_weight, vmax=1, vmin=0, s=50)
        plt.xlabel(r'\textbf{Time (seconds)}')
        plt.ylabel(r'\textbf{Space (mile)}')
        plt.title(r'\textbf{Congestion Weight}')
        plt.colorbar(label=r'\textbf{Weight}')
        plt.tight_layout()
        plt.savefig('congest.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=20)
        plt.scatter(st_df.time, -st_df.space, c=st_df.free_weight, vmax=1, vmin=0, s=50)
        plt.xlabel(r'\textbf{Time (seconds)}')
        plt.ylabel(r'\textbf{Space (mile)}')
        plt.title(r'\textbf{Free Weight}')
        plt.colorbar(label=r'\textbf{Weight}')
        plt.tight_layout()
        plt.savefig('free.png', dpi=300, bbox_inches='tight')
        plt.show()

    cong_weight_matrix = st_df.pivot(index='t', columns='x', values='cong_weight').values
    free_weight_matrix = st_df.pivot(index='t', columns='x', values='free_weight').values

    return cong_weight_matrix, free_weight_matrix


def smooth_speed_field(raw_data, cong_weight_matrix, free_weight_matrix):
    half_x_mat = int((cong_weight_matrix.shape[1] - 1) / 2)
    half_t_mat = int((cong_weight_matrix.shape[0] - 1) / 2)
    smooth_data = np.zeros(raw_data.shape)
    raw_data_w_bound = add_bounded_edges(raw_data, np.nan, half_t_mat, half_x_mat)
    for time_idx in tqdm(range(raw_data.shape[0])):
        for space_idx in range(raw_data.shape[1]):
            neighbour_matrix = raw_data_w_bound[time_idx:time_idx + 2 * half_t_mat + 1,
                               space_idx:space_idx + 2 * half_x_mat + 1]
            mask = pd.DataFrame(neighbour_matrix).notna().astype(int).values
            neighbour_fillna = pd.DataFrame(neighbour_matrix).fillna(0).values
            N_cong = np.sum(np.multiply(mask, cong_weight_matrix))
            N_free = np.sum(np.multiply(mask, free_weight_matrix))
            if N_cong == 0:
                v_cong = np.nan
            else:
                v_cong = np.sum(np.multiply(neighbour_fillna, cong_weight_matrix)) / N_cong
            if N_free == 0:
                v_free = np.nan
            else:
                v_free = np.sum(np.multiply(neighbour_fillna, free_weight_matrix)) / N_free
            if N_cong != 0 and N_free != 0:
                w = 0.5 * (1 + np.tanh((37.29 - min(v_cong, v_free)) / 12.43))
                v = w * v_cong + (1 - w) * v_free
            elif N_cong == 0:
                v = v_free
            elif N_free == 0:
                v = v_cong
            elif N_cong == 0 and N_free == 0:
                v = np.nan
            smooth_data[time_idx][space_idx] = v

    return smooth_data


def matrix_to_coordinates(matrix):
    """
    Converts a 2D matrix (numpy.array) into a list of coordinates with values.

    This function iterates through each element of a 2D matrix and
    creates a list of coordinates, where each coordinate is represented
    as a list containing the row index, column index, and the value at
    that position in the matrix.

    :param matrix: A numpy.array where each sublist represents a row in the matrix.
    :type matrix: numpy.array filled with float
    :return: A list of coordinates, where each coordinate is a list of [row_index, column_index, value].
    :rtype: list of list of float

    Example:
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        coords = matrix_to_coordinates(matrix)
        print(coords)  # Output: [[0, 0, 1], [0, 1, 2], [0, 2, 3], [1, 0, 4], [1, 1, 5], [1, 2, 6], [2, 0, 7], [2, 1, 8], [2, 2, 9]]
    """
    coordinates = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            coordinates.append([i, j, matrix[i][j]])
    return coordinates


def asm_data_w_x(processed_data, delta=0.12,tau=20, dx=0.02, dt=4, c_cong=13, c_free=-60,data_columns=['speed', 'occ', 'volume']):
    delta = delta
    dx = dx
    dt = dt
    c_cong = c_cong
    c_free = c_free
    t = abs(delta / c_cong / 2)
    x_mat = 2 * int(delta / dx / 2) + 1
    t_mat = int(t / dt * 3600) * 2 + 1
    matrix = np.zeros([x_mat, t_mat])
    matrix_df = pd.DataFrame(matrix)
    st_df = matrix_df.stack().reset_index()
    st_df.columns = ['x', 't', 'weight']
    st_df['time'] = dt * (st_df['t'] - int(t_mat / 2))
    st_df['space'] = dx * (st_df['x'] - int(x_mat / 2))
    # Define the variables speed and tau
    tau = tau
    # Function to fill weight based on the updated condition using speed and tau
    def fill_cong_weight(row):
        t_new = row['time'] - row['space'] / (c_cong / 3600)
        if abs(t_new) < tau / 2:
            return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))
        else:
            return 0

    def fill_free_weight(row):
        t_new = row['time'] - row['space'] / (c_free / 3600)
        if abs(t_new) < tau / 2:
            return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))
        else:
            return 0

    # Applying the function to the DataFrame
    st_df['cong_weight'] = st_df.apply(fill_cong_weight, axis=1)
    st_df['free_weight'] = st_df.apply(fill_free_weight, axis=1)
    cong_weight_matrix = st_df.pivot(index='t', columns='x', values='cong_weight').values
    free_weight_matrix = st_df.pivot(index='t', columns='x', values='free_weight').values
    half_x_mat = int((cong_weight_matrix.shape[1] - 1) / 2)
    half_t_mat = int((cong_weight_matrix.shape[0] - 1) / 2)
    # Assuming `processed_data` has columns for lane 1, 2, 3, and 4 speed, occupancy, and volume
    # lanes = [1, 2, 3, 4]
    # data_columns = ['speed', 'occ', 'volume']
    lanes = [1]
    data_columns = data_columns

    # Create the initial DataFrame for each lane and data type
    data = processed_data[
        ['milemarker', 'time_unix_fix'] + [f'lane{lane}_{col}' for lane in lanes for col in data_columns]]

    # Convert time_unix to datetime
    # Determine the range for the milemarker and time
    min_milemarker = data['milemarker'].min()
    max_milemarker = data['milemarker'].max()
    min_time_unix = data['time_unix_fix'].min()
    max_time_unix = data['time_unix_fix'].max()

    # Create a grid for the space (milemarkers) and time (in seconds)
    milemarkers = np.arange(min_milemarker, max_milemarker, dx)
    # Create a time range with a 4-second interval in Unix time
    time_range_unix = np.arange(min_time_unix, max_time_unix, dt)
    space_time_matrix_unix = pd.DataFrame(index=time_range_unix, columns=milemarkers)

    # Function to fill the space-time matrix for a given lane and data type
    def fill_space_time_matrix(data, lane, data_type):
        matrix = space_time_matrix_unix.copy()
        for index, row in data.iterrows():
            time_index = row['time_unix_fix']
            milemarker_index = row['milemarker']
            # Find the nearest time and milemarker grid points
            nearest_time = matrix.index.get_indexer([time_index], method='nearest')[0]
            nearest_milemarker = matrix.columns.get_indexer([milemarker_index], method='nearest')[0]
            # Assign the value to the nearest grid point
            matrix.iloc[nearest_time, nearest_milemarker] = row[f'lane{lane}_{data_type}']
        return matrix

    # Create smoothed data matrices for all lanes and data types
    smoothed_data = {}
    for lane in lanes:
        for data_type in data_columns:
            print(f'Processing lane {lane} {data_type}...')
            if data_type == 'speed':
                space_time_matrix = fill_space_time_matrix(data, lane, data_type)
                pre_smoothed_data = pd.DataFrame(space_time_matrix.values)

                # Perform smoothing
                pre_smoothed_data_w_bound = add_bounded_edges(pre_smoothed_data, np.nan, half_t_mat, half_x_mat)
                smooth_data = np.zeros(pre_smoothed_data.shape)
                for time_idx in tqdm(range(pre_smoothed_data.shape[0])):
                    for space_idx in range(pre_smoothed_data.shape[1]):
                        neighbour_matrix = pre_smoothed_data_w_bound[time_idx:time_idx + 2 * half_t_mat + 1,
                                           space_idx:space_idx + 2 * half_x_mat + 1]
                        mask = pd.DataFrame(neighbour_matrix).notna().astype(int).values
                        neighbour_fillna = pd.DataFrame(neighbour_matrix).fillna(0).values
                        N_cong = np.sum(np.multiply(mask, cong_weight_matrix))
                        N_free = np.sum(np.multiply(mask, free_weight_matrix))
                        if N_cong == 0:
                            v_cong = np.nan
                        else:
                            v_cong = np.sum(np.multiply(neighbour_fillna, cong_weight_matrix)) / N_cong
                        if N_free == 0:
                            v_free = np.nan
                        else:
                            v_free = np.sum(np.multiply(neighbour_fillna, free_weight_matrix)) / N_free
                        if N_cong != 0 and N_free != 0:
                            w = 0.5 * (1 + np.tanh((37.29 - min(v_cong, v_free)) / 12.43))
                            v = w * v_cong + (1 - w) * v_free
                        elif N_cong == 0:
                            v = v_free
                        elif N_free == 0:
                            v = v_cong
                        elif N_cong == 0 and N_free == 0:
                            v = np.nan
                        smooth_data[time_idx][space_idx] = v
                smoothed_data[(lane, data_type)] = smooth_data
            if data_type != 'speed':
                space_time_matrix = fill_space_time_matrix(data, lane, data_type)
                speed_matrix = fill_space_time_matrix(data, lane, 'speed')
                pre_smoothed_data = pd.DataFrame(space_time_matrix.values)
                pre_smoothed_data_speed = pd.DataFrame(speed_matrix.values)
                # Perform smoothing
                pre_smoothed_data_w_bound = add_bounded_edges(pre_smoothed_data, np.nan, half_t_mat, half_x_mat)
                pre_smoothed_data_speed_w_bound = add_bounded_edges(pre_smoothed_data_speed, np.nan, half_t_mat, half_x_mat)
                smooth_data = np.zeros(pre_smoothed_data.shape)
                for time_idx in tqdm(range(pre_smoothed_data.shape[0])):
                    for space_idx in range(pre_smoothed_data.shape[1]):
                        neighbour_matrix = pre_smoothed_data_w_bound[time_idx:time_idx + 2 * half_t_mat + 1,
                                           space_idx:space_idx + 2 * half_x_mat + 1]
                        neigbour_matrix_speed = pre_smoothed_data_speed_w_bound[time_idx:time_idx + 2 * half_t_mat + 1,
                                                  space_idx:space_idx + 2 * half_x_mat + 1]
                        mask = pd.DataFrame(neighbour_matrix).notna().astype(int).values
                        mask_speed = pd.DataFrame(neigbour_matrix_speed).notna().astype(int).values
                        neighbour_fillna = pd.DataFrame(neighbour_matrix).fillna(0).values
                        neighbour_fillna_speed = pd.DataFrame(neigbour_matrix_speed).fillna(0).values
                        N_cong = np.sum(np.multiply(mask, cong_weight_matrix))
                        N_free = np.sum(np.multiply(mask, free_weight_matrix))
                        N_cong_speed = np.sum(np.multiply(mask_speed, cong_weight_matrix))
                        N_free_speed = np.sum(np.multiply(mask_speed, free_weight_matrix))
                        if N_cong == 0:
                            v_cong = np.nan
                        else:
                            v_cong = np.sum(np.multiply(neighbour_fillna, cong_weight_matrix)) / N_cong
                            v_cong_speed = np.sum(np.multiply(neighbour_fillna_speed, cong_weight_matrix)) / N_cong_speed
                        if N_free == 0:
                            v_free = np.nan
                        else:
                            v_free = np.sum(np.multiply(neighbour_fillna, free_weight_matrix)) / N_free
                            v_free_speed = np.sum(np.multiply(neighbour_fillna_speed, free_weight_matrix)) / N_free_speed
                        if N_cong != 0 and N_free != 0:
                            w = 0.5 * (1 + np.tanh((37.29 - min(v_cong_speed, v_free_speed)) / 12.43))
                            v = w * v_cong + (1 - w) * v_free
                        elif N_cong == 0:
                            v = v_free
                        elif N_free == 0:
                            v = v_cong
                        elif N_cong == 0 and N_free == 0:
                            v = np.nan
                        smooth_data[time_idx][space_idx] = v
                smoothed_data[(lane, data_type)] = smooth_data
    # if there are still nan in smoothed data, fill them with the nearest value
    smoothed_data = {key: pd.DataFrame(value).fillna(method='ffill').fillna(method='bfill').values for key, value in smoothed_data.items()}
    # Convert the smoothed data to coordinates
    result_dfs = []
    for lane in lanes:
        for data_type in data_columns:
            smooth_data = smoothed_data[(lane, data_type)]
            smooth_data_df = pd.DataFrame(matrix_to_coordinates(smooth_data))
            smooth_data_df.columns = ['time_index', 'space_index', f'lane{lane}_{data_type}']
            smooth_data_df['unix_time'] = min_time_unix + smooth_data_df['time_index'] * dt + dt/2
            smooth_data_df['milemarker'] = min_milemarker + smooth_data_df['space_index'] * dx + dx/2
            result_dfs.append(smooth_data_df[['unix_time', 'milemarker', f'lane{lane}_{data_type}']])

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(result_dfs, axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df
