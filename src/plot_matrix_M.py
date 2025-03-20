import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import seaborn as sns

sns.set(style="whitegrid")
plt.rc('text')
plt.rc('font', family='serif')
LABELSIZE = 14
TICKSIZE = 12

def plot_matrix_mask(matrix, title="Matrix Sparsity Pattern"):
    """
    Plot the mask (sparsity pattern) of a matrix.
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to sparse matrix if it's a dense array
    if isinstance(matrix, np.ndarray):
        sparse_matrix = scipy.sparse.csr_matrix(matrix)
    else:
        sparse_matrix = matrix.copy()
        
    if not isinstance(sparse_matrix, scipy.sparse.coo_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    
    # Plot the non-zero elements
    plt.spy(sparse_matrix, markersize=0.5, color='blue')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Column Index", fontsize=14)
    plt.ylabel("Row Index", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"figures/boundaries/{title.replace(' ', '_').lower()}.pdf")
    plt.show()


def plot_detailed_matrix_view(matrix, title="Matrix Values", max_size=50):
    """
    Plot a more detailed view of the matrix, showing actual values.
    If matrix is large, only a subset will be shown.
    """
    # Convert to dense array for plotting
    if scipy.sparse.issparse(matrix):
        if matrix.shape[0] > max_size:
            # Extract a smaller portion for large matrices
            matrix_dense = matrix[:max_size, :max_size].toarray()
            title = f"{title} (First {max_size}x{max_size} Elements)"
        else:
            matrix_dense = matrix.toarray()
    else:
        if matrix.shape[0] > max_size:
            matrix_dense = matrix[:max_size, :max_size]
            title = f"{title} (First {max_size}x{max_size} Elements)"
        else:
            matrix_dense = matrix
    
    plt.figure(figsize=(12, 10))
    
    # Use a colormap that clearly shows zero vs non-zero
    cmap = plt.cm.viridis
    cmap.set_bad('white')
    
    # Plot matrix values
    im = plt.imshow(matrix_dense, cmap=cmap, interpolation='nearest')
    
    # Add a colorbar
    plt.colorbar(im, label="Matrix Value")
    
    plt.title(title, fontsize=16)
    plt.xlabel("Column Index", fontsize=14)
    plt.ylabel("Row Index", fontsize=14)
    
    # Add grid lines to separate cells for small matrices
    if matrix_dense.shape[0] <= 20:
        plt.grid(which='both', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"figures/boundaries/{title.replace(' ', '_').lower()}.pdf")
    plt.show()


def visualize_matrix_structure(solver, plot_values=True):
    """
    Visualize the structure of the matrix A in the MembraneSolver.
    """
    # Create directory for figures if it doesn't exist
    import os
    os.makedirs("figures", exist_ok=True)
    
    # Plot the sparsity pattern
    plot_matrix_mask(solver.A, f"{solver.shape.capitalize()} Membrane Matrix Sparsity")
    
    # Plot the actual values if requested
    if plot_values:
        plot_detailed_matrix_view(solver.A, f"{solver.shape.capitalize()} Membrane Matrix Values")
    
    if scipy.sparse.issparse(solver.A):
        nnz = solver.A.nnz
        size = solver.A.shape[0] * solver.A.shape[1]
        sparsity = (1 - nnz/size) * 100
        print(f"Matrix shape: {solver.A.shape}")
        print(f"Number of non-zero elements: {nnz}")
        print(f"Sparsity: {sparsity:.2f}% (percentage of zeros)")
    else:
        nnz = np.count_nonzero(solver.A)
        size = solver.A.shape[0] * solver.A.shape[1]
        sparsity = (1 - nnz/size) * 100
        print(f"Matrix shape: {solver.A.shape}")
        print(f"Number of non-zero elements: {nnz}")
        print(f"Sparsity: {sparsity:.2f}% (percentage of zeros)")