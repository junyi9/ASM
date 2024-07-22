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

def generate_weight_matrices(delta=0.12, dx=0.02, dt=4, c_cong=12.5, c_free=-45, tau=20, plot=True):
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


def smooth_speed_field_nan(raw_data, cong_weight_matrix, free_weight_matrix):
    half_x_mat = int((cong_weight_matrix.shape[1] - 1) / 2)
    half_t_mat = int((cong_weight_matrix.shape[0] - 1) / 2)
    smooth_data = raw_data
    raw_data_w_bound = add_bounded_edges(raw_data, np.nan, half_t_mat, half_x_mat)
    # Find the indices of NaN values
    nan_indices = np.where(np.isnan(raw_data))

    # Extract the unique row indices
    unique_row_indices = np.unique(nan_indices[0])

    # Initialize a list to store unique column indices for each unique row index
    unique_col_indices_per_row = []

    # Iterate over each unique row index
    for row_idx in unique_row_indices:
        # Get the columns where NaN values are located for the current row
        col_indices = nan_indices[1][nan_indices[0] == row_idx]

        # Get unique column indices for the current row
        unique_col_indices = np.unique(col_indices)

        # Append to the list
        unique_col_indices_per_row.append(unique_col_indices)

    for row_idx, col_indices in zip(unique_row_indices, unique_col_indices_per_row):
        for col_idx in col_indices:
            neighbour_matrix = raw_data_w_bound[row_idx:row_idx + 2 * half_t_mat + 1,
                               col_idx:col_idx + 2 * half_x_mat + 1]
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
            smooth_data[row_idx][col_idx] = v

    return smooth_data
