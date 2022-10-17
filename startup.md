# TO GET ON COMPUTE NODE (FROM LOGIN NODE)
screen -S interact
screen -r interact
interact -n 4 -m 12g -t 01:00:00

# BEFORE YOU START
git pull
export PYTHONPATH="${PYTHONPATH}:/users/pvankatw/emulator"

# TO GET TO DATA DIRECTORY
cd /users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/