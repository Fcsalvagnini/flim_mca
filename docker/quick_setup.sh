#!/bin/bash

echo "Setting up FLIM-MCA Docker environment..."

# Create .env file with current user's UID and GID
cat > .env << EOF
USER_NAME=$(whoami)
USER_ID=$(id -u)
GROUP_ID=$(id -g)
EOF

echo "Created .env file with your user permissions"
echo "Starting FLIM-MCA container..."

docker-compose -f docker-compose.public.yml up -d

echo ""
echo "Container started! Access it with:"
echo "docker exec -it flim_mca bash"