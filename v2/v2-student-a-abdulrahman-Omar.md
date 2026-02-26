# Student A Submission — Abdulrahman Omar

**Assignment:** MLOps Assignment 1 — Reproducibility Challenge (Version 2)  
**Role:** Student A (Code Author)  
**Date:** February 26, 2026

---

## What I Built

A **Simple Generative Adversarial Network (GAN)** implemented in pure NumPy that learns to generate synthetic tabular data from a CSV dataset.

### Files Provided to Student B

| File | Description |
|------|-------------|
| `gan_model.py` | GAN training script (pure NumPy, no TensorFlow) |
| `gan_data.csv` | Dataset with 500 rows × 5 numerical features |

### What I Did NOT Provide (Intentionally)

- ❌ No `requirements.txt`
- ❌ No documentation or README
- ❌ No version information
- ❌ No random seeds
- ❌ No virtual environment setup instructions

---

## Model Architecture

### Generator

```
Input (latent_dim=32) → Dense(64) → LeakyReLU(0.2)
                      → Dense(128) → LeakyReLU(0.2)
                      → Dense(5, sigmoid)
```

### Discriminator

```
Input (5 features) → Dense(128) → LeakyReLU(0.2)
                   → Dense(64) → LeakyReLU(0.2)
                   → Dense(1, sigmoid)
```

### Training Configuration

- **Optimizer:** Manual SGD (learning rate = 0.0002)
- **Loss:** Binary Cross-Entropy
- **Epochs:** 500
- **Batch Size:** 32
- **Latent Dimension:** 32

---

## My Training Output

```
Epoch 0   | D Loss: 1.3864 | D Acc Real:  0.0% | D Acc Fake: 100.0% | G Loss: 0.6933
Epoch 100 | D Loss: 1.3864 | D Acc Real:  0.0% | D Acc Fake: 100.0% | G Loss: 0.6933
Epoch 200 | D Loss: 1.3863 | D Acc Real:  0.0% | D Acc Fake: 100.0% | G Loss: 0.6933
Epoch 300 | D Loss: 1.3863 | D Acc Real:  0.0% | D Acc Fake: 100.0% | G Loss: 0.6934
Epoch 400 | D Loss: 1.3863 | D Acc Real:  0.0% | D Acc Fake: 100.0% | G Loss: 0.6934

Final Discriminator Accuracy on Real: 1.00%
Final Discriminator Accuracy on Fake: 100.00%
```

---

## Libraries Used (Not Documented for Student B)

| Library | Purpose |
|---------|---------|
| `numpy` | All neural network operations, weight init, forward/backward pass |
| `pandas` | Loading CSV data |
| `matplotlib` | Plotting training loss curves |

**No versions were pinned.** Student B has no way to know which versions I used.

---

## Known Reproducibility Issues in My Code

1. **No random seeds** — `np.random.seed()` is never called, so every run produces different weight initializations, noise vectors, and minibatch sampling.

2. **No `requirements.txt`** — Student B must guess all dependencies by reading the import statements and installing them one by one.

3. **No version pinning** — NumPy 2.x has breaking changes from 1.x; different versions may produce different numerical results.

4. **Hardcoded CSV path** — `pd.read_csv("gan_data.csv")` assumes the file is in the current working directory.

5. **No model saving** — The trained weights exist only in memory; lost when the script finishes.

6. **No logging or error handling** — Only `print()` statements; failures produce raw tracebacks.

---

## What I Should Have Done (MLOps Best Practices)

```bash
# 1. Set random seed at the top
np.random.seed(42)

# 2. Pin dependencies
pip freeze > requirements.txt

# 3. Save model weights
np.savez("gan_weights.npz", g_W1=g_W1, g_b1=g_b1, ...)

# 4. Containerize with Dockerfile

# 5. Use proper logging instead of print()

# 6. Version the data with checksums or DVC
```

---

## Conclusion

This submission intentionally demonstrates **bad MLOps practices** to highlight the reproducibility challenges that arise when dependencies are not documented, environments are not controlled, randomness is not managed, and code is not packaged for portability.
