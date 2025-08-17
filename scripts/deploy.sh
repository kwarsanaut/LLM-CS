#!/bin/bash

# Deployment script for IndoBERT Document Customer Service
# Supports Docker, Docker Compose, and Kubernetes deployment

set -e

echo "🚀 IndoBERT Document Customer Service Deployment"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Default configuration
DEPLOYMENT_TYPE="docker-compose"
ENVIRONMENT="production"
BUILD_IMAGE=true
PUSH_IMAGE=false
REGISTRY=""
TAG="latest"
NAMESPACE="default"
DOMAIN=""
SSL_ENABLED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --no-build)
            BUILD_IMAGE=false
            shift
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --ssl)
            SSL_ENABLED=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [OPTIONS]

Options:
    --type TYPE         Deployment type (docker-compose, kubernetes, docker)
    --env ENV           Environment (production, staging, development)
    --no-build          Skip building Docker image
    --push              Push image to registry
    --registry REGISTRY Docker registry URL
    --tag TAG           Docker image tag (default: latest)
    --namespace NS      Kubernetes namespace (default: default)
    --domain DOMAIN     Domain for ingress
    --ssl               Enable SSL/TLS
    -h, --help          Show this help message

Examples:
    $0 --type docker-compose --env production
    $0 --type kubernetes --namespace indobert --domain api.example.com --ssl
    $0 --type docker --no-build --tag v1.0.0
EOF
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Configuration based on environment
case $ENVIRONMENT in
    production)
        print_status "Configuring for production environment"
        ;;
    staging)
        print_status "Configuring for staging environment"
        ;;
    development)
        print_status "Configuring for development environment"
        ;;
    *)
        print_error "Unknown environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Pre-deployment checks
print_header "Running pre-deployment checks..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
required_files=("Dockerfile" "docker-compose.yml" "requirements.txt" "config/model_config.yaml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Required file not found: $file"
        exit 1
    fi
done

print_status "Pre-deployment checks passed"

# Build Docker image
if [ "$BUILD_IMAGE" = true ]; then
    print_header "Building Docker image..."
    
    IMAGE_NAME="indobert-document-cs"
    if [ -n "$REGISTRY" ]; then
        FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"
    else
        FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
    fi
    
    docker build -t "$FULL_IMAGE_NAME" --target production .
    
    if [ $? -eq 0 ]; then
        print_status "Docker image built successfully: $FULL_IMAGE_NAME"
    else
        print_error "Docker image build failed"
        exit 1
    fi
    
    # Push image if requested
    if [ "$PUSH_IMAGE" = true ] && [ -n "$REGISTRY" ]; then
        print_header "Pushing image to registry..."
        docker push "$FULL_IMAGE_NAME"
        print_status "Image pushed to registry"
    fi
fi

# Deploy based on type
case $DEPLOYMENT_TYPE in
    docker-compose)
        deploy_docker_compose
        ;;
    kubernetes)
        deploy_kubernetes
        ;;
    docker)
        deploy_docker
        ;;
    *)
        print_error "Unknown deployment type: $DEPLOYMENT_TYPE"
        exit 1
        ;;
esac

print_status "Deployment completed successfully!"

# Deployment functions
deploy_docker_compose() {
    print_header "Deploying with Docker Compose..."
    
    # Create environment-specific compose file
    COMPOSE_FILE="docker-compose.yml"
    if [ "$ENVIRONMENT" = "production" ]; then
        COMPOSE_FILE="docker-compose.prod.yml"
    fi
    
    # Check if compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_warning "Compose file $COMPOSE_FILE not found, using default docker-compose.yml"
        COMPOSE_FILE="docker-compose.yml"
    fi
    
    # Stop existing containers
    print_status "Stopping existing containers..."
    docker-compose -f "$COMPOSE_FILE" down || true
    
    # Start services
    print_status "Starting services..."
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f "$COMPOSE_FILE" up -d --remove-orphans
    else
        docker-compose -f "$COMPOSE_FILE" up -d
    fi
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "✅ Application is running and healthy"
        print_status "🌐 Access the application at: http://localhost:8000"
    else
        print_warning "⚠️  Application may not be fully ready yet"
        print_status "📋 Check logs with: docker-compose logs -f"
    fi
}

deploy_kubernetes() {
    print_header "Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Create Kubernetes manifests
    create_kubernetes_manifests
    
    # Apply manifests
    print_status "Applying Kubernetes manifests..."
    kubectl apply -f k8s/ -n "$NAMESPACE"
    
    # Wait for deployment
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/indobert-document-cs -n "$NAMESPACE"
    
    # Get service URL
    if [ -n "$DOMAIN" ]; then
        PROTOCOL="http"
        if [ "$SSL_ENABLED" = true ]; then
            PROTOCOL="https"
        fi
        print_status "🌐 Application available at: ${PROTOCOL}://${DOMAIN}"
    else
        SERVICE_URL=$(kubectl get svc indobert-document-cs -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -n "$SERVICE_URL" ]; then
            print_status "🌐 Application available at: http://${SERVICE_URL}:8000"
        else
            print_status "📋 Get service URL with: kubectl get svc -n $NAMESPACE"
        fi
    fi
}

deploy_docker() {
    print_header "Deploying with Docker..."
    
    # Stop existing container
    docker stop indobert-document-cs || true
    docker rm indobert-document-cs || true
    
    # Run container
    docker run -d \
        --name indobert-document-cs \
        --restart unless-stopped \
        -p 8000:8000 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        -e ENVIRONMENT="$ENVIRONMENT" \
        "$FULL_IMAGE_NAME"
    
    # Health check
    sleep 10
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "✅ Application is running and healthy"
        print_status "🌐 Access the application at: http://localhost:8000"
    else
        print_warning "⚠️  Application may not be fully ready yet"
        print_status "📋 Check logs with: docker logs indobert-document-cs"
    fi
}

create_kubernetes_manifests() {
    print_status "Creating Kubernetes manifests..."
    
    mkdir -p k8s
    
    # Deployment manifest
    cat > k8s/deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: indobert-document-cs
  labels:
    app: indobert-document-cs
spec:
  replicas: 2
  selector:
    matchLabels:
      app: indobert-document-cs
  template:
    metadata:
      labels:
        app: indobert-document-cs
    spec:
      containers:
      - name: indobert-document-cs
        image: ${FULL_IMAGE_NAME}
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "${ENVIRONMENT}"
        - name: PYTHONPATH
          value: "/app"
        - name: TOKENIZERS_PARALLELISM
          value: "false"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: indobert-data-pvc
      - name: models-volume
        persistentVolumeClaim:
          claimName: indobert-models-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: indobert-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: indobert-document-cs
spec:
  selector:
    app: indobert-document-cs
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: indobert-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: indobert-models-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: indobert-logs-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
EOF
    
    # Ingress manifest (if domain specified)
    if [ -n "$DOMAIN" ]; then
        cat > k8s/ingress.yaml << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: indobert-document-cs
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
$(if [ "$SSL_ENABLED" = true ]; then
    echo "    cert-manager.io/cluster-issuer: \"letsencrypt-prod\""
fi)
spec:
$(if [ "$SSL_ENABLED" = true ]; then
    cat << EOFTLS
  tls:
  - hosts:
    - ${DOMAIN}
    secretName: indobert-tls
EOFTLS
fi)
  rules:
  - host: ${DOMAIN}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: indobert-document-cs
            port:
              number: 80
EOF
    fi
    
    print_status "Kubernetes manifests created in k8s/ directory"
}

# Rollback function
rollback() {
    print_header "Rolling back deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            docker-compose down
            ;;
        kubernetes)
            kubectl delete -f k8s/ -n "$NAMESPACE" || true
            ;;
        docker)
            docker stop indobert-document-cs || true
            docker rm indobert-document-cs || true
            ;;
    esac
    
    print_status "Rollback completed"
}

# Cleanup function
cleanup() {
    print_header "Cleaning up temporary files..."
    rm -rf k8s/
    print_status "Cleanup completed"
}

# Signal handlers
trap cleanup EXIT
trap 'print_error "Deployment interrupted"; rollback; exit 1' INT TERM
