#!/bin/bash
# Simple script to build and serve documentation locally

set -e

echo "Building documentation..."
cd "$(dirname "$0")"
sphinx-build -b html source build/html

echo "Documentation built successfully!"
echo "Starting local server at http://localhost:8000"
echo "Press Ctrl+C to stop the server"

cd build/html
python3 -m http.server 8000
