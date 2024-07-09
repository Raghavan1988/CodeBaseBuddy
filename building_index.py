import os
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex
import sys

# Load the advanced SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

# Retrieve command-line arguments
name = sys.argv[1]
GITHUB_PATH = sys.argv[2]

def get_files(path):
    """
    Recursively retrieves all .py, .sh, and .java files from the given directory.
    
    Args:
        path (str): The root directory to start the search.
    
    Returns:
        list: A list of file paths.
    """
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if ".py" in file or ".sh" in file or ".java" in file:
                files.append(os.path.join(r, file))
    return files

def get_file_embeddings(path):
    """
    Generates embeddings for the contents of a given file.
    
    Args:
        path (str): The file path.
    
    Returns:
        numpy.ndarray: The embedding vector or None if an error occurs.
    """
    try:
        text = get_file_contents(path)
        ret = model.encode(text)
        return ret
    except Exception as e:
        print(f"Error in embedding file: {path} - {e}")
        return None

def get_file_contents(path):
    """
    Reads and returns the contents of a given file.
    
    Args:
        path (str): The file path.
    
    Returns:
        str: The contents of the file.
    """
    with open(path, 'r') as f:
        return f.read()

# Main script execution
print(name)
print(GITHUB_PATH)

# Get all relevant files from the specified directory
files = get_files(GITHUB_PATH)
print(f"Number of files found: {len(files)}")

embeddings_dict = {}  # Dictionary to store embeddings with file paths
index_map = {}  # Dictionary to map index to file paths
i = 0  # Counter for processed files

# Process each file to generate embeddings
for file in files:
    e = get_file_embeddings(file)
    if e is None:
        continue
    embeddings_dict[file] = e
    index_map[i] = file
    i += 1
    if i % 100 == 0:
        print(f"Number of files processed: {i}")

# Get the embedding dimension from the model
embedding_dim = model.get_sentence_embedding_dimension()

# Create an Annoy index with the appropriate dimensions and distance metric
t = AnnoyIndex(embedding_dim, 'angular')

# Add embeddings to the Annoy index
for idx, (file, embedding) in enumerate(embeddings_dict.items()):
    t.add_item(idx, embedding)

# Build the Annoy index
t.build(len(files))

# Save the Annoy index to a file
index_filename = name + "_mpnet.ann"
t.save(index_filename)

# Save the index-to-filepath mapping to a text file
with open('index_map_' + name + '.txt', 'w') as f:
    for idx, path in index_map.items():
        f.write(f'{idx}\t{path}\n')

# Output the results
print("Index created: " + index_filename)
print("Number of files indexed: " + str(len(embeddings_dict)))
