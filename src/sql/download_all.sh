#!/bin/bash

# Define an array of download scripts and their corresponding load-production scripts
declare -A scripts
scripts=(
    ["download-publication-venues.py"]="load-production-publication-venues.py"
)

# Base directory containing the scripts
base_dir="$HOME/phd/s2ag-corpus/src"

# Function to run a script in a new screen session and wait for it to complete
run_script() {
    script=$1
    screen_name="${script%.*}" # Use script name without extension as screen session name
    screen -dmS "$screen_name" bash -c "python3 $base_dir/$script > /tmp/$screen_name.log 2>&1"
    echo "Started $script in screen session $screen_name"
    # Wait for the script to finish
    screen -S "$screen_name" -X stuff 'exit\n'
    while screen -list | grep -q "$screen_name"; do
        sleep 1
    done
    echo "Completed $script"
}

# Iterate over each download script and run its corresponding load-production script
for download_script in "${!scripts[@]}"; do
    load_script=${scripts[$download_script]}
    run_script "$download_script"
    run_script "$load_script"
done

echo "All scripts are now running sequentially in separate screen sessions."
