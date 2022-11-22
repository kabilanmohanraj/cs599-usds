# Remote Sensing with Ray

This project aims at downloading and analyzing TIF files from NASA repositories so as to convert them to analyzable 
format and run models on it

# Commands to start running locally
1. docker build -t cs599-rsr-image .
2. docker run --network="host" -d cs599-rsr-image:latest
3. Run 'docker ps' to get the container id of the ray instance
4. docker exec -t -i <container-id> bash
5. You can then run 'python application.py --urls=1' to make sure you ray instance is working as expected

# Tracing
For tracing purpose you will find functions with comment 'TODO: Trace this function'. Those functions need to be traced
as part of task 3. 

# Commands for running on multiple servers

Make sure to install docker on all your servers
Follow instructions from here: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04

## Head Node

1. sudo docker build -t cs599-rsr-image .
2. sudo docker run -d --network="host" --shm-size=2.48gb cs599-rsr-image:latest

## Worker Node
1. sudo docker build -t cs599-rsr-image .
2. sudo docker run --network="host" --shm-size=2.48gb -d cs599-rsr-image:latest worker <IP Address of the head node>

## Starting application
You should run the below commands on your head node:
1. Run 'sudo docker ps' to get the container id of the ray instance
2. sudo docker exec -t -i <container-id> bash
3. You can then run 'python application.py --urls=1' to make sure you ray instance is working as expected