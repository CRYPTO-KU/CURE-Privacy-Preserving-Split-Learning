#!/bin/bash

# Create src.txt with directory structure
tree -I 'data|*.exe|*.o|*.a|*.so|*.dylib|*.dll|.DS_Store|.git|AGENTS' > src.txt

echo -e "\n\n# ===== SOURCE CODE FILES =====\n" >> src.txt

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

# Extract configuration files
extract_file "go.mod" "go.mod"
extract_file "go.sum" "go.sum"
extract_file ".gitignore" ".gitignore"

# Extract all Go files recursively
echo -e "\n# Finding all Go files..." >> src.txt
find . -name "*.go" -type f | while read -r go_file; do
    # Skip files in data directory
    if [[ "$go_file" != *"/data/"* ]]; then
        extract_file "$go_file" "$go_file"
    fi
done

# Extract all README files
echo -e "\n# Finding all README files..." >> src.txt
find . -name "README.md" -type f | while read -r readme_file; do
    # Skip files in data directory
    if [[ "$readme_file" != *"/data/"* ]]; then
        extract_file "$readme_file" "$readme_file"
    fi
done

echo "Source code extraction completed to src.txt" 