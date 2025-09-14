#!/bin/bash
# Run this script to generate a single-header include for Ollama.hpp that combines the header-only libraries in ./include

# Remove the includes to other libraries in ollama.hpp
sed -e "47d" -e "54d" -e "62d" "include/ollama.hpp" > "no_includes_ollama.hpp"

# Combine the headers into one file
{ cat "include/json.hpp"; echo -e "\n\n"; cat "include/httplib.h"; echo -e "\n\n"; cat "include/Base64.h"; echo -e "\n\n"; cat "no_includes_ollama.hpp"; } > "singleheader/ollama.hpp"

# Clean up
rm "no_includes_ollama.hpp"

echo "Header-only library created in singleheader/ollama.hpp"
