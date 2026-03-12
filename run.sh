#!/bin/sh
# Run main.py and forward all output to PID 1's stdout/stderr
# so Docker's logging driver captures it even when invoked via `docker exec`.
exec python -u main.py "$@" >> /proc/1/fd/1 2>> /proc/1/fd/2
