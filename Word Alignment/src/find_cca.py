from sklearn.decomposition import PCA
import numpy as np

def perform_pca(data, components):
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def calculate_G(B1, B2, X1, X2, D):
    M = X1.T @ B1.T + X2.T @ B2.T
    U, S, VT = np.linalg.svd(M)
    GT = U[:, :D] @ VT

    return GT


def optimize_cca(GT, X1, X2):

    B1 = GT.T @ np.linalg.pinv(X1)
    B2 = GT.T @ np.linalg.pinv(X2)

    return B1, B2

def find_cca(X_1, X_2, D, MAX_CCA_ITER = 100, eps = 1e-4):

    GT = perform_pca(X_1.T, D)

    previous_error = 0
    error_vs_iter = []

    for i in range(MAX_CCA_ITER):

        B1, B2 = optimize_cca(GT, X_1, X_2)
        GT = calculate_G(B1, B2, X_1, X_2, D)

        error = np.linalg.norm(
            np.matmul(B1, X_1) - np.matmul(B2, X_2), ord="fro"
        )

        print(
            f"Iteration: {i+1}, Error: {error}"
        )
        error_vs_iter.append( error )

        if abs(previous_error - error) < eps and i>3:
            print(f"Final Error{abs(previous_error - error)}")
            break

        previous_error = error
        
    return B1, B2, error


def generate_permutation(mm, num_cols_per, mat=None):
    if mat is None:
        permuted_matrix = np.eye(mm)
    else:
        permuted_matrix = np.copy(mat)
        mm = mat.shape[1]
        
    cols_index_to_swap = np.random.choice(np.arange(mm), int(num_cols_per*mm), replace=False)
    cols_known = [i for i in range(mm) if i not in cols_index_to_swap]
    permute_order = np.random.permutation(cols_index_to_swap)
    permuted_matrix[:,cols_index_to_swap] = permuted_matrix[:,permute_order]
    
    return permuted_matrix, cols_known