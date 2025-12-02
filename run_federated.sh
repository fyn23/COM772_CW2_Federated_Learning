#!/bin/bash
# Launch script for FEMNIST federated learning
# Starts server + 20 clients in parallel

set -e  # Exit on error

# Configuration
NUM_CLIENTS=20
NUM_ROUNDS=50
EPOCHS=5
STRATEGY="fedprox"  # Options: fedavg, fedprox
MU=0.1
SERVER_ADDRESS="127.0.0.1:8080"
DATA_TYPE="niid"  # Options: "niid" or "iid"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' 

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FEMNIST Federated Learning Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Clients:   ${GREEN}${NUM_CLIENTS}${NC}"
echo -e "Rounds:    ${GREEN}${NUM_ROUNDS}${NC}"
echo -e "Server:    ${GREEN}${SERVER_ADDRESS}${NC}"
echo -e "Data type: ${GREEN}${DATA_TYPE}${NC}"
echo -e "Epochs:    ${GREEN}${EPOCHS}${NC}"
echo -e "Strategy:  ${GREEN}${STRATEGY}${NC}"
echo ""

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    pkill -P $$ 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Create logs directory
mkdir -p logs

# Generate run name with format: clients_rounds_datatype_timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${NUM_CLIENTS}clients_${NUM_ROUNDS}rounds_${DATA_TYPE}_${TIMESTAMP}"

# Start server in background
echo -e "${BLUE}Starting server...${NC}"
python3 model/server.py \
    --rounds ${NUM_ROUNDS} \
    --min_clients ${NUM_CLIENTS} \
    --server_address ${SERVER_ADDRESS} \
    --strategy ${STRATEGY} \
    --mu ${MU} \
    --run_name ${RUN_NAME} \
    > logs/server.log 2>&1 &

SERVER_PID=$!
echo -e "Server PID: ${GREEN}${SERVER_PID}${NC}"

# Wait for server to be ready
echo -e "${YELLOW}Waiting for server to start...${NC}"
sleep 3

# Start clients in background
echo -e "${BLUE}Starting ${NUM_CLIENTS} clients...${NC}"
for i in $(seq 1 ${NUM_CLIENTS}); do
    python3 model/client.py \
        --server_address ${SERVER_ADDRESS} \
        --epochs ${EPOCHS} \
        --data_type ${DATA_TYPE} \
        > logs/client_${i}.log 2>&1 &
    
    CLIENT_PID=$!
    echo -e "  Client ${i} started (PID: ${CLIENT_PID})"
    
    # Small delay to avoid overwhelming the system
    sleep 0.2
done

echo -e "\n${GREEN}All clients started!${NC}"
echo -e "${YELLOW}Training in progress...${NC}"
echo -e "Monitor logs in ${BLUE}logs/${NC} directory"
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop all processes\n"

# Wait for server to complete
wait ${SERVER_PID}

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Training complete!${NC}"
echo -e "${GREEN}========================================${NC}"

# Optional: show final server log
echo -e "\n${BLUE}Final server output:${NC}"
tail -n 20 logs/server.log

# Cleanup any remaining client processes
pkill -P $$ 2>/dev/null || true
