#!/bin/bash

# Create .env file with current user's UID and GID
echo "Creating .env file with current user permissions..."

cat > .env << EOF
USER_NAME=flim_ca
USER_ID=$(id -u)
GROUP_ID=$(id -g)
EOF

echo "Created .env file:"
cat .env

echo ""
echo "Now you can build and run with:"
echo "docker-compose build"
echo "docker-compose up -d"
echo ""
echo "Or rebuild if needed:"
echo "docker-compose build --no-cache"