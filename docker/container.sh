#!/usr/bin/env bash

# Copyright (c) 2024, Robotis Lab Project Developers.
# All rights reserved.
#
# Based on Isaac Lab container management script

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get source directory
export ROBOTISLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
export DOCKER_DIR="${ROBOTISLAB_PATH}/docker"

#==
# Helper functions
#==

# print the usage description
print_help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<args>]"
    echo -e "\nRobotis Lab Docker Container Management Script"
    echo -e "\noptional arguments:"
    echo -e "  -h, --help           Display this help message."
    echo ""
    echo -e "commands:"
    echo -e "  build                Build the docker image for Robotis Lab"
    echo -e "  start                Start the docker container"
    echo -e "  enter                Enter the running docker container"
    echo -e "  stop                 Stop the docker container"
    echo -e "  clean                Remove the docker container and image"
    echo -e "  logs                 Show logs from the container"
    echo ""
}

# Load environment variables
load_env() {
    if [ -f "${DOCKER_DIR}/.env.base" ]; then
        set -a
        source "${DOCKER_DIR}/.env.base"
        set +a
        echo "[INFO] Loaded environment from .env.base"
    else
        echo "[ERROR] .env.base file not found in ${DOCKER_DIR}"
        exit 1
    fi
}

# Build docker image
build_image() {
    echo "[INFO] Building Robotis Lab docker image..."
    cd "${DOCKER_DIR}"
    docker compose build robotis-lab-base
    echo "[INFO] Build complete!"
}

# Start docker container
start_container() {
    echo "[INFO] Starting Robotis Lab docker container..."
    cd "${DOCKER_DIR}"
    
    # Check if container is already running
    if docker ps | grep -q "robotis-lab-base"; then
        echo "[INFO] Container is already running"
        return 0
    fi
    
    # Check if container exists but is stopped
    if docker ps -a | grep -q "robotis-lab-base"; then
        echo "[INFO] Starting existing container..."
        docker compose start robotis-lab-base
    else
        echo "[INFO] Creating and starting new container..."
        docker compose up -d robotis-lab-base
    fi
    
    echo "[INFO] Container started successfully!"
    echo "[INFO] Use './docker/container.sh enter' to access the container"
}

# Enter running container
enter_container() {
    echo "[INFO] Entering Robotis Lab docker container..."
    
    # Check if container is running
    if ! docker ps | grep -q "robotis-lab-base"; then
        echo "[ERROR] Container is not running. Start it first with './docker/container.sh start'"
        exit 1
    fi
    
    docker exec -it robotis-lab-base${DOCKER_NAME_SUFFIX} /bin/bash
}

# Stop container
stop_container() {
    echo "[INFO] Stopping Robotis Lab docker container..."
    cd "${DOCKER_DIR}"
    docker compose stop robotis-lab-base
    echo "[INFO] Container stopped"
}

# Clean up container and image
clean_docker() {
    echo "[INFO] Cleaning up Robotis Lab docker resources..."
    cd "${DOCKER_DIR}"
    
    read -p "This will remove the container and image. Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down robotis-lab-base
        docker rmi robotis-lab-base${DOCKER_NAME_SUFFIX}:latest || true
        echo "[INFO] Cleanup complete"
    else
        echo "[INFO] Cleanup cancelled"
    fi
}

# Show container logs
show_logs() {
    echo "[INFO] Showing Robotis Lab container logs..."
    cd "${DOCKER_DIR}"
    docker compose logs -f robotis-lab-base
}

#==
# Main
#==

# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2
    print_help
    exit 1
fi

# Load environment variables
load_env

# pass the arguments
case "$1" in
    build)
        build_image
        ;;
    start)
        start_container
        ;;
    enter)
        enter_container
        ;;
    stop)
        stop_container
        ;;
    clean)
        clean_docker
        ;;
    logs)
        show_logs
        ;;
    -h|--help)
        print_help
        exit 0
        ;;
    *)
        echo "[Error] Invalid command: $1"
        print_help
        exit 1
        ;;
esac

echo ""
echo "[INFO] Command completed successfully!"
