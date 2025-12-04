# CURE_lib Dockerfile
# Multi-stage build for privacy-preserving deep learning with HE
# PoPETs 2026 Artifact - Pinned versions for reproducibility

# =============================================================================
# Stage 1: Builder
# =============================================================================
# Pin Go version for reproducibility (golang:1.23.3-alpine3.19)
FROM golang:1.23.3-alpine3.19 AS builder

# Install build dependencies
RUN apk add --no-cache git make

# Set working directory
WORKDIR /app

# Copy go mod files first for better caching
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build all binaries
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /out/cure-bench ./cmd/benchmarks && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /out/cure-train ./cmd/train && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /out/cure-server ./cmd/server && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /out/cure-client ./cmd/client && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /out/cure-infer ./cmd/infer

# =============================================================================
# Stage 2: Runtime
# =============================================================================
# Pin Alpine version for reproducibility
FROM alpine:3.19.1 AS runtime

# Install CA certificates for any HTTPS calls
RUN apk add --no-cache ca-certificates

# Create non-root user for security
RUN adduser -D -g '' cureuser

# Set working directory
WORKDIR /app

# Copy binaries from builder
COPY --from=builder /out/ /app/

# Copy data directory structure (empty, user provides data)
RUN mkdir -p /app/data/mnist

# Change ownership
RUN chown -R cureuser:cureuser /app

# Switch to non-root user
USER cureuser

# Default command
ENTRYPOINT ["/app/cure-bench"]
CMD ["--help"]

# =============================================================================
# Stage 3: Development (optional, for development use)
# =============================================================================
# Pin Go version for reproducibility
FROM golang:1.23.3-alpine3.19 AS dev

# Install development tools
RUN apk add --no-cache git make bash

# Set working directory
WORKDIR /app

# Copy source
COPY . .

# Install dependencies
RUN go mod download

# Default command for development
CMD ["go", "test", "./..."]
