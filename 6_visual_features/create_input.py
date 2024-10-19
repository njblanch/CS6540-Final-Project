import string

def generate_combinations(start, end, output_file):
    # Get all lowercase letters
    letters = string.ascii_lowercase
    
    # Convert start and end to lists of indices (e.g., 'xaa' -> [23, 0, 0])
    start_idx = [letters.index(c) for c in start]
    end_idx = [letters.index(c) for c in end]
    
    # Helper function to increment the indices
    def increment(idx):
        if idx[2] < 25:
            idx[2] += 1
        else:
            idx[2] = 0
            if idx[1] < 25:
                idx[1] += 1
            else:
                idx[1] = 0
                idx[0] += 1
    
    # Open file
    with open(output_file, 'w') as f:
        # All combinations
        current_idx = start_idx[:]
        while current_idx <= end_idx:
            # Convert current indices to letters
            current_folder = ''.join(letters[i] for i in current_idx)
            
            f.write(current_folder + '\n')
            
            if current_idx == end_idx:
                break
            
            increment(current_idx)

start = 'xaa'
end = 'xdz'
output_file = './input_file.txt'

generate_combinations(start, end, output_file)
