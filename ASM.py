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

def fill_cong_weight(row):
    t_new = row['time'] - row['space'] / (c_cong/3600)
    if abs(t_new) < tau/2:
        return np.exp(-(abs(t_new)/tau+abs(row['space'])/delta))
    else:
        return 0

def fill_free_weight(row):
    t_new = row['time'] - row['space'] / (c_free/3600)
    if abs(t_new) < tau/2:
        return np.exp(-(abs(t_new)/tau+abs(row['space'])/delta))
    else:
        return 0

  
