#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo -e "${GREEN}Starting Recommendation System CLI...${NC}"

## parse arguments and run cli.py

if [ "$#" -gt 0 ]; then
    python3 cli.py "$@"
else
    echo -e "${RED}No arguments provided. Please provide necessary arguments to run the CLI.${NC}"
    echo -e "${RED}Necessary Argument is --model with values ["knn", "bsl", "nn"]${NC}"
    exit 1
fi

echo -e "${GREEN}Recommendation System CLI exited successfully.${NC}"