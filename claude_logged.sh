#!/bin/bash
mkdir -p logs
LOGFILE="logs/claude_$(date +%Y%m%d_%H%M%S).txt"
script -f "$LOGFILE"
