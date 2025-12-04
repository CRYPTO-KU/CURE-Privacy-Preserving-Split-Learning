# CURE_lib Makefile
# Privacy-preserving deep learning with Homomorphic Encryption

.PHONY: all build test clean release docker help

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOTEST=$(GOCMD) test
GOMOD=$(GOCMD) mod
BINARY_DIR=bin
LDFLAGS=-ldflags="-w -s"

# Binary names
BINARIES=cure-bench cure-train cure-server cure-client cure-infer

# Default target
all: build

## help: Show this help message
help:
	@echo "CURE_lib - Privacy-Preserving Deep Learning with HE"
	@echo ""
	@echo "Usage:"
	@echo "  make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       Build all binaries (default)"
	@echo "  build     Build all binaries for current platform"
	@echo "  test      Run all unit tests"
	@echo "  test-v    Run all unit tests with verbose output"
	@echo "  test-short Run quick tests only"
	@echo "  bench     Run benchmarks"
	@echo "  release   Cross-compile for Linux and macOS (amd64/arm64)"
	@echo "  docker    Build Docker image"
	@echo "  clean     Remove built binaries"
	@echo "  deps      Download dependencies"
	@echo "  tidy      Tidy go.mod"
	@echo "  lint      Run linter (requires golangci-lint)"
	@echo "  help      Show this help message"

## build: Build all binaries for current platform
build: deps
	@mkdir -p $(BINARY_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/cure-bench ./cmd/benchmarks
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/cure-train ./cmd/train
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/cure-server ./cmd/server
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/cure-client ./cmd/client
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/cure-infer ./cmd/infer
	@echo "Build complete: $(BINARY_DIR)/"
	@ls -la $(BINARY_DIR)/

## test: Run all unit tests
test:
	$(GOTEST) ./nn/... ./core/... ./tensor/... ./utils/... -count=1

## test-v: Run all unit tests with verbose output
test-v:
	$(GOTEST) -v ./nn/... ./core/... ./tensor/... ./utils/... -count=1

## test-short: Run quick tests only (skip long-running tests)
test-short:
	$(GOTEST) -short ./nn/... ./core/... ./tensor/... ./utils/... -count=1

## bench: Run benchmarks
bench:
	$(GOTEST) -bench=. -benchmem ./nn/... ./core/...

## release: Cross-compile for multiple platforms
release: deps
	@mkdir -p $(BINARY_DIR)/release
	@echo "Building for Linux amd64..."
	@for bin in $(BINARIES); do \
		GOOS=linux GOARCH=amd64 CGO_ENABLED=0 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/release/$$bin-linux-amd64 ./cmd/$$(echo $$bin | sed 's/cure-//'); \
	done
	@echo "Building for Linux arm64..."
	@for bin in $(BINARIES); do \
		GOOS=linux GOARCH=arm64 CGO_ENABLED=0 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/release/$$bin-linux-arm64 ./cmd/$$(echo $$bin | sed 's/cure-//'); \
	done
	@echo "Building for macOS amd64..."
	@for bin in $(BINARIES); do \
		GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/release/$$bin-darwin-amd64 ./cmd/$$(echo $$bin | sed 's/cure-//'); \
	done
	@echo "Building for macOS arm64..."
	@for bin in $(BINARIES); do \
		GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/release/$$bin-darwin-arm64 ./cmd/$$(echo $$bin | sed 's/cure-//'); \
	done
	@echo "Release binaries built in $(BINARY_DIR)/release/"
	@ls -la $(BINARY_DIR)/release/

## docker: Build Docker image
docker:
	docker build -t cure-lib:latest .
	docker build --target dev -t cure-lib:dev .

## docker-run: Run benchmarks in Docker
docker-run:
	docker run --rm cure-lib:latest --logN=13 --cores=4

## docker-test: Run tests in Docker
docker-test:
	docker run --rm cure-lib:dev go test ./nn/... ./core/... ./tensor/... ./utils/... -short

## clean: Remove built binaries
clean:
	rm -rf $(BINARY_DIR)
	rm -f cure-bench cure-train cure-server cure-client cure-infer

## deps: Download dependencies
deps:
	$(GOMOD) download

## tidy: Tidy go.mod
tidy:
	$(GOMOD) tidy

## lint: Run linter
lint:
	@which golangci-lint > /dev/null || (echo "Install golangci-lint: https://golangci-lint.run/usage/install/" && exit 1)
	golangci-lint run ./...

## fmt: Format code
fmt:
	$(GOCMD) fmt ./...

## vet: Run go vet
vet:
	$(GOCMD) vet ./nn/... ./core/... ./tensor/... ./utils/... ./cmd/...
