#!/bin/bash
# PostgreSQL Test Database Setup Script
# This script creates a dedicated test database for rag-tester E2E tests

set -e  # Exit on error

echo "=== PostgreSQL Test Database Setup ==="
echo ""

# Step 1: Create postgres user and testdb database
echo "Step 1: Creating postgres user and testdb database..."
psql -d postgres << 'SQL'
-- Create the postgres user with superuser privileges
CREATE USER postgres WITH PASSWORD 'postgres' SUPERUSER;

-- Create the dedicated test database
CREATE DATABASE testdb OWNER postgres;
SQL

echo "✓ User and database created"
echo ""

# Step 2: Install pgvector extension
echo "Step 2: Installing pgvector extension..."
psql -U postgres -d testdb << 'SQL'
-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
SQL

echo "✓ pgvector extension installed"
echo ""

# Step 3: Verify setup
echo "Step 3: Verifying setup..."
psql -U postgres -d testdb -c "\du postgres"
psql -U postgres -d testdb -c "\l testdb"
psql -U postgres -d testdb -c "\dx"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run PostgreSQL E2E tests, use:"
echo "  POSTGRESQL_URL=\"postgresql://postgres:postgres@localhost:5432/testdb\" make test ARGS=\"tests/e2e/test_postgresql.py -v\""
echo ""
echo "Or set the environment variable permanently:"
echo "  export POSTGRESQL_URL=\"postgresql://postgres:postgres@localhost:5432/testdb\""
echo "  make test ARGS=\"tests/e2e/test_postgresql.py -v\""
