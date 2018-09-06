#/bin/sh

python fight.py simple-1 1 12345678 ws://localhost:8080 simple &
python fight.py simple-2 2 12345678 ws://localhost:8080 simple &
python fight.py simple-3 3 12345678 ws://localhost:8080 simple &
