# Setup Instructions and Useful Info

## TO GET ON COMPUTE NODE (FROM LOGIN NODE)

screen -S interact  
screen -r interact  
interact -n 4 -m 12g -t 01:00:00  

## BEFORE YOU START

git pull  
export PYTHONPATH="${PYTHONPATH}:/users/pvankatw/emulator"  

## TO GET TO DATA DIRECTORY

cd /users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/  

## Tensorboard Partition

(256, 128, 64, 32, 16, 1|64, 128, 32, 16, 8, 1|64, 20, 1|128, 64, 32, 1|256, 128, 64, 32, 16, 8, 4, 1)
(dataset1|dataset2|dataset3|dataset4|all_columns|dataset5)
