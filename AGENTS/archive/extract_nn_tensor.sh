#!/bin/bash

# Clear the src.txt file first
> src.txt

echo -e "# ===== NN AND TENSOR SOURCE CODE EXTRACTION =====\n" >> src.txt

# Function to extract a file
extract_file() {
    local file_path="$1"
    local display_name="$2"
    
    if [ -f "$file_path" ]; then
        echo -e "\n# $display_name\n" >> src.txt
        echo "cat > $file_path << 'EOF'" >> src.txt
        cat "$file_path" >> src.txt
        echo -e "\nEOF\n" >> src.txt
    fi
}

# Extract all files from nn/ directory recursively
echo -e "\n# ===== NN DIRECTORY FILES =====\n" >> src.txt

# Extract nn/ files (excluding subdirectories for now)
for file in nn/*.go nn/*.md; do
    if [ -f "$file" ]; then
        extract_file "$file" "$file"
    fi
done

# Extract nn/layers/ files
echo -e "\n# ===== NN/LAYERS DIRECTORY FILES =====\n" >> src.txt
for file in nn/layers/*.go; do
    if [ -f "$file" ]; then
        extract_file "$file" "$file"
    fi
done

# Extract nn/bench/ files
echo -e "\n# ===== NN/BENCH DIRECTORY FILES =====\n" >> src.txt
for file in nn/bench/*.go; do
    if [ -f "$file" ]; then
        extract_file "$file" "$file"
    fi
done

# Extract all files from tensor/ directory
echo -e "\n# ===== TENSOR DIRECTORY FILES =====\n" >> src.txt
for file in tensor/*.go tensor/*.md; do
    if [ -f "$file" ]; then
        extract_file "$file" "$file"
    fi
done

echo "NN and Tensor source code extraction completed to src.txt" 