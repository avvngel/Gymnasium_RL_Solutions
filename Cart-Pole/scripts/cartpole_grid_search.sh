#!/usr/bin/env bash

# Create an empty file to store commands
commands_file="commands.txt"
rm -f "$commands_file" && touch "$commands_file"

results_file="grid_search_results_epsilon_greedy_gamma_almost_one.txt"
rm -f "$results_file" && touch "$results_file"

# Generate commands for each value of N and store them in commands_file
for i in {2..10..2}; do
	echo "python ../src/main.py --N $i --n_epochs 10 --epsilon .1 --gamma .8 > results_N_${i}_10_epochs_.1_epsilon_99_gamma.txt" >> "$commands_file"
done

# Use GNU parallel to run the commands in commands_file
parallel < "$commands_file"

# After parallel execution, extract and append the results to grid_search_results.txt
for i in {2..10..2}; do

    # Extract rewards and print them along with N and the epoch they correspond to.
    awk -v N="$i" -F': ' '/Average reward:/ {
        count++; # Keep track of the number of regex matches.
        epoch = count * 2; # Calculate the corresponding epoch.
        printf "   %-12s%-15s%-15s\n", N, epoch, $2
    }' results_N_${i}_10_epsilon_epochs_99_gamma.txt >> $results_file
done
