#!/bin/bash

# Wait for tritontest service to be ready
until docker-compose exec tritontest curl -s http://localhost:8000/v2/health/ready | grep -q "\"ready\": true"; do
  >&2 echo "tritontest is unavailable - sleeping"
  sleep 1
done

>&2 echo "tritontest is up - waiting for 10 seconds before starting triton_client"
#sleep 10
#>&2 echo "Starting triton_client"
#docker-compose exec triton_client bash -c "python3 run.py"

