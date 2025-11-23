# Makefile for Miyraa API
# Provides common commands for development and deployment

.PHONY: help install test lint build run clean docker-build docker-run docker-stop monitoring

# Default target
help:
	@echo "Miyraa API - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make install          - Install dependencies"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Run linters"
	@echo "  make run              - Run API locally"
	@echo "  make clean            - Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-stop      - Stop Docker container"
	@echo "  make docker-logs      - View Docker logs"
	@echo ""
	@echo "Docker Compose:"
	@echo "  make compose-up       - Start all services"
	@echo "  make compose-down     - Stop all services"
	@echo "  make monitoring       - Start with monitoring stack"
	@echo ""
	@echo "Production:"
	@echo "  make deploy-k8s       - Deploy to Kubernetes"
	@echo "  make health-check     - Check API health"

# Development
install:
	pip install --upgrade pip
	pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=100
	black src/ tests/ --check

format:
	black src/ tests/

run:
	python -m uvicorn src.api.main:app --reload --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf build dist *.egg-info

# Docker
docker-build:
	docker build -t miyraa:latest .

docker-run:
	docker run -d \
		-p 8000:8000 \
		-e LOG_LEVEL=info \
		-e LOG_FORMAT=json \
		-v $$(pwd)/logs:/app/logs \
		--name miyraa-api \
		miyraa:latest

docker-stop:
	docker stop miyraa-api || true
	docker rm miyraa-api || true

docker-logs:
	docker logs -f miyraa-api

docker-shell:
	docker exec -it miyraa-api /bin/bash

# Docker Compose
compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f

compose-restart:
	docker-compose restart miyraa-api

monitoring:
	docker-compose --profile monitoring up -d

# Production
deploy-k8s:
	kubectl apply -f k8s/deployment.yaml

health-check:
	@curl -f http://localhost:8000/health && echo "✅ API is healthy" || echo "❌ API is unhealthy"

ready-check:
	@curl -f http://localhost:8000/ready && echo "✅ API is ready" || echo "❌ API is not ready"

metrics:
	@curl -s http://localhost:8000/metrics

# CI/CD
ci-test:
	pytest tests/ -v --cov=src --cov-report=xml

ci-build:
	docker build -t miyraa:$${CI_COMMIT_SHA} .

ci-push:
	docker tag miyraa:$${CI_COMMIT_SHA} registry.example.com/miyraa:$${CI_COMMIT_SHA}
	docker push registry.example.com/miyraa:$${CI_COMMIT_SHA}
