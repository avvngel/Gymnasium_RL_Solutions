#!/usr/bin/env bash

# set PYTHONPATH to include the modules directory
export PYTHONPATH="${PYTHONPATH}:/home/angelleal/Documents/CSE370/Gymnasium_RL_Solutions/Cart-Pole/modules"

# Configure DISPLAY for X server communication from WSL
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
