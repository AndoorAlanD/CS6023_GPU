import random

def generate_matrix_input(filename, H, W, C, min_val=0, max_val=1001):
    with open(filename, 'w') as f:
        # Write the size of the matrix
        f.write(f"{H} {W} {C}\n")
        
        # Generate and write the matrix values
        for _ in range(H * C):
            val = [str(random.randint(min_val, max_val)) for _ in range(W)]
            f.write(" ".join(val) + "\n")

def generate_filter_input(filename, R, S, C, K, min_val=-101, max_val=1001):
    with open(filename, 'a') as f:  # Append to the same file
        # Write the size of the filter
        f.write(f"{C} {R} {S} {K}\n")
        

        # Generate and write the filter values
        for _ in range(R * C * K):
            val = [str(random.randint(min_val, max_val)) for _ in range(S)]
            f.write(" ".join(val) + "\n")

if __name__ == "__main__":
    filename = ""
    H = 512  
    W = 1024  
    C = 10     
    R = 5
    S = 7
    K = 11   

    generate_matrix_input(filename, H, W, C)
    generate_filter_input(filename, R, S, C, K)
    print(f"Matrix input file '{filename}' generated successfully.")
