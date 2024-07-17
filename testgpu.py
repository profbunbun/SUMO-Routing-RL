import torch
import time

def test_large_matrix_multiplication(device):
    # Set the size of the matrices (e.g., 10000x10000)
    size = 10000
    print(f"Testing large matrix multiplication for size {size}x{size}")

    # Create random matrices
    matrix1 = torch.randn(size, size, device=device)
    matrix2 = torch.randn(size, size, device=device)

    # Start the timer
    start_time = time.time()

    # Perform matrix multiplication
    result = torch.matmul(matrix1, matrix2)

    # Stop the timer
    elapsed_time = time.time() - start_time
    print(f"Matrix multiplication completed in {elapsed_time:.2f} seconds")

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! Performing a large matrix multiplication test.")
    device = torch.device("cuda")
    test_large_matrix_multiplication(device)
else:
    print("CUDA is not available. Please check your installation.")
