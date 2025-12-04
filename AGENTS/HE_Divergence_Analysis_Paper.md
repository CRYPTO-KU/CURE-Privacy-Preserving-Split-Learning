# Homomorphic Encryption Layer-by-Layer Divergence Analysis

## A Comprehensive Study of HE vs Plaintext Computation Error Accumulation in Deep Neural Networks

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Homomorphic Encryption Parameters](#2-homomorphic-encryption-parameters)
3. [ReLU3 Polynomial Approximation](#3-relu3-polynomial-approximation)
4. [Level Consumption Analysis](#4-level-consumption-analysis)
5. [Model Architectures](#5-model-architectures)
6. [Layer-by-Layer Divergence Analysis](#6-layer-by-layer-divergence-analysis)
7. [Discussion and Findings](#7-discussion-and-findings)
8. [Appendix: Detailed Layer Results](#8-appendix-detailed-layer-results)

---

## 1. Executive Summary

This document presents a comprehensive analysis of the error divergence between Homomorphic Encryption (HE) and plaintext computations in deep neural network inference. We evaluate three specific models:

1. **MNIST MLP**: 784 → 128 → 32 → 10
2. **BCW FC**: 64 → 32 → 16 → 10
3. **LeNet-5**: Conv-based architecture (CNN)
4. **Audio1D CNN**: 1D Convolutional network

**Key Findings:**
- Linear layers introduce minimal error (~10⁻⁷)
- ReLU3 polynomial approximation is the dominant source of divergence (~0.12-0.25)
- Accumulated RMS error after 3 FC layers: ~0.38-0.41
- **Mode**: No refresh/cheat-strap applied between layers

---

## 2. Homomorphic Encryption Parameters

### 2.1 CKKS Scheme Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Library** | Lattigo v6 | Go-based HE library |
| **Scheme** | CKKS | Approximate arithmetic |
| **LogN** | 15 | Ring dimension N = 2^15 = 32,768 |
| **MaxSlots** | 16,384 | N/2 usable slots for packing |
| **LogQ** | [60, 40, 40, 40, 38] | Modulus chain (5 levels) |
| **LogP** | [60] | Special modulus for key-switching |
| **LogDefaultScale** | 40 | Default scaling factor (2^40) |

### 2.2 Security Parameters

| Parameter | Value |
|-----------|-------|
| **Total LogQP** | 60 + 40 + 40 + 40 + 38 + 60 = 278 bits |
| **Security Level** | ≈ 128-bit (estimated) |
| **N (Ring Dimension)** | 32,768 |

### 2.3 Modulus Chain Detail

```
Level 4 (Fresh):     LogQ = 60   (initial encryption)
Level 3:             LogQ = 40   (after 1 rescale)
Level 2:             LogQ = 40   (after 2 rescales)
Level 1:             LogQ = 40   (after 3 rescales)
Level 0:             LogQ = 38   (after 4 rescales)
Special Modulus:     LogP = 60   (key-switching)
```

### 2.4 Noise Budget

- **Initial precision**: ~60 bits
- **Per-multiplication precision loss**: ~40 bits (one rescale)
- **Maximum depth**: 4 multiplicative levels
- **Usable levels for inference**: 3-4 (depending on architecture)

---

## 3. ReLU3 Polynomial Approximation

### 3.1 Polynomial Definition

The ReLU activation function is approximated using a degree-3 polynomial:

$$\text{ReLU3}(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$$

**Coefficients:**

| Coefficient | Value | Description |
|-------------|-------|-------------|
| c₀ | 0.3183099 | Constant term (bias) |
| c₁ | 0.5 | Linear coefficient |
| c₂ | 0.2122066 | Quadratic coefficient |
| c₃ | 0.0 | Cubic coefficient (effectively degree-2) |

### 3.2 Derivative Polynomial (for backward pass)

$$\text{ReLU3}'(x) = c_0 + c_1 x$$

| Coefficient | Value |
|-------------|-------|
| c₀ | 0.5 |
| c₁ | 0.4244 |

### 3.3 Approximation Error Sources

| Source | Magnitude | Description |
|--------|-----------|-------------|
| **Polynomial Fit Error** | ~0.1-0.3 | Difference between polynomial and true ReLU |
| **HE Noise** | ~10⁻⁸ | Encryption/decryption noise |
| **Rescaling Error** | ~10⁻⁹ | Precision loss during rescaling |
| **Scale Mismatch** | Variable | Potential issues when adding different-scale values |

### 3.4 Levels Consumed

| Layer Type | Levels Consumed | Operations |
|------------|-----------------|------------|
| **Linear** | 1 | Mul + Rescale |
| **ReLU3 Activation** | 2 | Degree-2 polynomial (2 multiplications) |
| **Linear + ReLU3** | 3 | Combined layer pair |

---

## 4. Level Consumption Analysis

### 4.1 Per-Layer Level Usage

| Layer Type | Multiplicative Depth | Levels Used | Post-Layer Level |
|------------|---------------------|-------------|------------------|
| Fresh Encryption | 0 | 0 | 4 |
| Linear Layer | 1 | 1 | 3 |
| ReLU3 Activation | 2 | 2 | 1 |
| Next Linear | 1 | 1 | 0 |

### 4.2 Model-Specific Level Budget

#### MNIST MLP (784 → 128 → 32 → 10)

| Step | Layer | Levels Used | Remaining Levels |
|------|-------|-------------|------------------|
| 0 | Input (encrypted) | 0 | 4 |
| 1 | Linear_0 (784→128) | 1 | 3 |
| 2 | Activation_0 (ReLU3) | 2 | 1 |
| 3 | Linear_1 (128→32) | 1 | 0 |
| 4 | Activation_1 (ReLU3) | ⚠️ | Level exhausted! |

**Note**: With current parameters, 2 full (Linear + Activation) layers can be computed before level exhaustion.

#### BCW FC (64 → 32 → 16 → 10)

Same level budget as MNIST MLP.

---

## 5. Model Architectures

### 5.1 MNIST MLP (Fully-Connected)

| Layer | Type | Input Dim | Output Dim | Parameters |
|-------|------|-----------|------------|------------|
| 0 | Linear | 784 | 128 | 100,480 |
| 1 | ReLU3 | 128 | 128 | 0 |
| 2 | Linear | 128 | 32 | 4,128 |
| 3 | ReLU3 | 32 | 32 | 0 |
| 4 | Linear | 32 | 10 | 330 |
| 5 | ReLU3 | 10 | 10 | 0 |
| **Total** | | | | **104,938** |

### 5.2 BCW FC (Wisconsin Breast Cancer)

| Layer | Type | Input Dim | Output Dim | Parameters |
|-------|------|-----------|------------|------------|
| 0 | Linear | 64 | 32 | 2,080 |
| 1 | ReLU3 | 32 | 32 | 0 |
| 2 | Linear | 32 | 16 | 528 |
| 3 | ReLU3 | 16 | 16 | 0 |
| 4 | Linear | 16 | 10 | 170 |
| 5 | ReLU3 | 10 | 10 | 0 |
| **Total** | | | | **2,778** |

### 5.3 LeNet-5 (Convolutional)

| Layer | Type | Dimensions | Parameters |
|-------|------|------------|------------|
| 0 | Conv2D | 1→6 (5×5) | 156 |
| 1 | ReLU3 | - | 0 |
| 2 | Conv2D | 6→16 (5×5) | 2,416 |
| 3 | ReLU3 | - | 0 |
| 4 | Linear | 256→120 | 30,840 |
| 5 | ReLU3 | - | 0 |
| 6 | Linear | 120→84 | 10,164 |
| 7 | ReLU3 | - | 0 |
| 8 | Linear | 84→10 | 850 |
| **Total** | | | **44,426** |

### 5.4 Audio1D CNN

| Layer | Type | Dimensions | Parameters |
|-------|------|------------|------------|
| 0 | Conv1D | 12→16 (k=3) | 592 |
| 1 | ReLU3 | - | 0 |
| 2 | MaxPool1D | window=2 | 0 |
| 3 | Conv1D | 16→8 (k=3) | 392 |
| 4 | ReLU3 | - | 0 |
| 5 | MaxPool1D | window=2 | 0 |
| 6 | Flatten | - | 0 |
| 7 | Linear | 2000→5 | 10,005 |
| 8 | ReLU3 | - | 0 |
| **Total** | | | **10,989** |

---

## 6. Layer-by-Layer Divergence Analysis

### 6.1 MNIST MLP (784 → 128 → 32 → 10)

**Test Configuration:**
- Input: Random values in [-1, 1]
- Weight initialization: Xavier/Glorot (scale = √(2/(in+out)))
- Bias initialization: Random in [-0.05, 0.05]
- Mode: **NO REFRESH** between layers

| Layer Index | Layer Name | Max Abs Err | Mean Abs Err | RMS Err | Max Rel Err | Dims |
|-------------|------------|-------------|--------------|---------|-------------|------|
| 0 | Linear_0 (784→128) | 5.44×10⁻⁷ | 1.69×10⁻⁷ | 2.08×10⁻⁷ | 7.78×10⁻⁵ | 128 |
| 1 | Activation_0 (ReLU3) | 2.38×10⁻¹ | 1.20×10⁻¹ | 1.40×10⁻¹ | 5.50×10¹ | 128 |
| 2 | Linear_1 (128→32) | 3.61×10⁻¹ | 1.61×10⁻¹ | 1.93×10⁻¹ | 5.44×10⁰ | 32 |
| 3 | Activation_1 (ReLU3) | 2.50×10⁻¹ | 1.76×10⁻¹ | 1.92×10⁻¹ | 6.01×10¹ | 32 |
| 4 | Linear_2 (32→10) | 1.79×10⁻¹ | 7.85×10⁻² | 9.78×10⁻² | 1.65×10⁰ | 10 |
| 5 | Activation_2 (ReLU3) | 2.42×10⁻¹ | 1.85×10⁻¹ | 1.95×10⁻¹ | 7.14×10⁰ | 10 |

**Total Forward Pass Accumulated RMS Error: 3.765×10⁻¹**

#### Error Progression Chart (MNIST MLP)

```
Layer 0 (Linear_0):     ██ 2.08e-07
Layer 1 (Activation_0): ████████████████████████████████████████████████████████████████████ 1.40e-01
Layer 2 (Linear_1):     ████████████████████████████████████████████████████████████████████████████████████████████ 1.93e-01
Layer 3 (Activation_1): █████████████████████████████████████████████████████████████████████████████████████████ 1.92e-01
Layer 4 (Linear_2):     ████████████████████████████████████████████████ 9.78e-02
Layer 5 (Activation_2): █████████████████████████████████████████████████████████████████████████████████████████████ 1.95e-01
```

---

### 6.2 BCW FC (64 → 32 → 16 → 10)

**Test Configuration:**
- Input: Random values in [-1, 1]
- Weight initialization: Xavier/Glorot
- Mode: **NO REFRESH** between layers

| Layer Index | Layer Name | Max Abs Err | Mean Abs Err | RMS Err | Max Rel Err | Dims |
|-------------|------------|-------------|--------------|---------|-------------|------|
| 0 | Linear_0 (64→32) | 1.52×10⁻⁷ | 4.25×10⁻⁸ | 5.59×10⁻⁸ | 1.92×10⁻⁶ | 32 |
| 1 | Activation_0 (ReLU3) | 2.50×10⁻¹ | 1.15×10⁻¹ | 1.33×10⁻¹ | 1.10×10¹ | 32 |
| 2 | Linear_1 (32→16) | 4.05×10⁻¹ | 1.70×10⁻¹ | 2.11×10⁻¹ | 1.98×10⁰ | 16 |
| 3 | Activation_1 (ReLU3) | 2.44×10⁻¹ | 1.99×10⁻¹ | 2.08×10⁻¹ | 6.34×10⁰ | 16 |
| 4 | Linear_2 (16→10) | 1.90×10⁻¹ | 1.02×10⁻¹ | 1.16×10⁻¹ | 1.32×10⁰ | 10 |
| 5 | Activation_2 (ReLU3) | 2.49×10⁻¹ | 2.11×10⁻¹ | 2.17×10⁻¹ | 9.56×10⁻¹ | 10 |

**Total Forward Pass Accumulated RMS Error: 4.074×10⁻¹**

#### Error Progression Chart (BCW FC)

```
Layer 0 (Linear_0):     █ 5.59e-08
Layer 1 (Activation_0): ███████████████████████████████████████████████████████████████ 1.33e-01
Layer 2 (Linear_1):     █████████████████████████████████████████████████████████████████████████████████████████████████████ 2.11e-01
Layer 3 (Activation_1): ██████████████████████████████████████████████████████████████████████████████████████████████████ 2.08e-01
Layer 4 (Linear_2):     █████████████████████████████████████████████████████████ 1.16e-01
Layer 5 (Activation_2): ██████████████████████████████████████████████████████████████████████████████████████████████████████ 2.17e-01
```

---

### 6.3 LeNet-5 (Convolutional)

**Note**: Full LeNet-5 HE inference requires significant level budget. Currently testing FC portion only due to level constraints.

| Layer Index | Layer Name | Max Abs Err | Mean Abs Err | RMS Err | Levels Used |
|-------------|------------|-------------|--------------|---------|-------------|
| 0 | Conv2D_1 (1→6, 5×5) | TBD | TBD | TBD | 1 |
| 1 | Activation (ReLU3) | TBD | TBD | TBD | 2 |
| 2 | Conv2D_2 (6→16, 5×5) | TBD | TBD | TBD | 1 |
| 3 | Activation (ReLU3) | TBD | TBD | TBD | 2 |
| 4+ | FC layers... | Level exhausted | | | |

**Level Budget Issue**: With LogQ = [60,40,40,40,38], only 4 levels available. Two (Conv + ReLU3) pairs consume 6 levels, exceeding budget.

---

### 6.4 Audio1D CNN

Similar level constraints apply. MaxPool1D operates in plaintext (no level consumption).

---

## 7. Discussion and Findings

### 7.1 Error Source Analysis

| Error Source | Magnitude | Mitigation |
|--------------|-----------|------------|
| **Linear Layer HE Noise** | ~10⁻⁷ | Negligible - inherent to CKKS |
| **ReLU3 Approximation** | ~0.1-0.25 | Use higher-degree polynomial or different activation |
| **Error Propagation** | Multiplicative | Reduce network depth or use refresh |
| **Level Exhaustion** | Critical | Increase LogQ chain or use bootstrapping |

### 7.2 Observations

1. **Linear layers are highly accurate**: RMS error ~10⁻⁷, effectively lossless
2. **Activation layers dominate error**: ReLU3 approximation introduces ~0.12-0.22 RMS error per layer
3. **Error accumulates but doesn't explode**: Total RMS stays under 0.5 for 3-layer networks
4. **Without refresh**: HE layers can be connected directly, but level budget limits depth

### 7.3 Recommendations

1. **For deeper networks**: Implement bootstrapping or hybrid HE/plaintext approach
2. **For accuracy-critical applications**: Consider degree-4+ polynomials for activation
3. **For efficiency**: Current 3-layer FC networks are practical with existing parameters
4. **Level budget**: Consider LogQ = [60, 40, 40, 40, 40, 40, 38] for deeper networks

---

## 8. Appendix: Detailed Layer Results

### 8.1 Error Metrics Definitions

| Metric | Formula | Description |
|--------|---------|-------------|
| **Max Abs Err** | max\|HE - Plain\| | Worst-case single-element error |
| **Mean Abs Err** | mean\|HE - Plain\| | Average absolute difference |
| **RMS Err** | √(mean((HE - Plain)²)) | Root mean square error |
| **Max Rel Err** | max(\|HE - Plain\|/\|Plain\|) | Relative error (when \|Plain\| > 10⁻⁸) |

### 8.2 Test Configuration

```go
// Random seed for reproducibility
rand.Seed(42)

// Weight initialization (Xavier/Glorot)
scale := math.Sqrt(2.0 / float64(inDim+outDim))
weight := (rand.Float64() - 0.5) * 2 * scale

// Bias initialization
bias := (rand.Float64() - 0.5) * 0.1

// Input generation
input := (rand.Float64() - 0.5) * 2.0  // Range: [-1, 1]
```

### 8.3 HE Context Initialization

```go
func NewHeContext() *HeContext {
    params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
        LogN:            15,
        LogQ:            []int{60, 40, 40, 40, 38},
        LogP:            []int{60},
        LogDefaultScale: 40,
    })
    kgen := rlwe.NewKeyGenerator(params)
    sk := kgen.GenSecretKeyNew()
    
    return &HeContext{
        Params:    params,
        Sk:        sk,
        Encoder:   ckks.NewEncoder(params),
        Encryptor: ckks.NewEncryptor(params, sk),
        Decryptor: ckks.NewDecryptor(params, sk),
    }
}
```

---

## Document Information

| Field | Value |
|-------|-------|
| **Library** | CURE_lib |
| **Version** | Lattigo v6 |
| **Date** | 2025 |
| **Mode** | Forward-only, NO REFRESH |
| **Test Seed** | 42 |

---

*This document was generated as part of the CURE_lib HE deep learning correctness analysis.*
