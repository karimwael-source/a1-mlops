# Student B's Reproducibility Report

**Assignment:** MLOps — Reproducibility Challenge  
**Date:** February 26, 2026  
**Student A's Code:** `gan_model.py` (Simple GAN using pure NumPy for tabular data generation)  
**Data File:** `gan_data.csv` (500 rows × 5 features)

---

## 1. How many commands did I have to run before it worked?

I had to run **4 commands** before the code executed successfully:

1. `python3 -m venv .venv` — Created a virtual environment.
2. `pip install numpy` — Missing dependency, no requirements.txt provided.
3. `pip install pandas` — Missing dependency.
4. `pip install matplotlib` — Missing dependency (for plotting).

Student A provided **no `requirements.txt`**, so I had to guess every dependency manually by reading import statements and fixing `ModuleNotFoundError` one by one.

---

## 2. What libraries were missing? Did version mismatches cause errors?

**Missing libraries:** `numpy`, `pandas`, `matplotlib` — none were documented.

**Version mismatch issues encountered:**

- **NumPy version:** Student A did not specify which NumPy version was used. NumPy 2.x introduced breaking changes from 1.x (e.g., deprecated type aliases like `np.float` removed, changes to random number generation internals). This can cause subtle numerical differences even with identical code.
- **Matplotlib backend:** The code uses `matplotlib.use('Agg')` which works headless, but without it `plt.show()` would crash on a server. Student A did handle this, but it wasn't documented.
- **Pandas API:** Minor API changes across versions could affect CSV loading behavior (e.g., dtype inference changes).

---

## 3. Did the model produce the same result? If not, why?

**No, the model did NOT produce the same results.** The reasons are:

1. **No random seed was set.** Student A's code does not call `np.random.seed()`. Every run produces different weight initializations, different noise vectors, and different minibatch sampling — leading to completely different training trajectories.

2. **Data shuffling is non-deterministic.** The line `np.random.randint(0, X.shape[0], batch_size)` samples random indices each epoch without a fixed seed, so the discriminator sees different batches every run.

3. **Weight initialization is random.** `np.random.randn()` is used to initialize all network weights without a seed, so every run starts from a different point.

4. **No model checkpointing.** There is no saved model file, so I cannot load Student A's exact trained weights to compare.

**To fix this**, Student A should have added:

```python
np.random.seed(42)
```

---

## 4. If this had to run on a server at 3:00 AM, would it survive?

**Probably not.** Here's why:

| Issue | Impact |
|-------|--------|
| No `requirements.txt` or `environment.yml` | Server has no way to install correct dependencies automatically |
| No version pinning | A `pip install numpy` today may install a different version than Student A used |
| No random seeds | Every run produces different results — impossible to validate or audit |
| No error handling | If data file is missing or corrupted, the script crashes with raw traceback |
| No logging | No way to check what happened at 3 AM — just print statements to stdout |
| No model saving | The trained model is lost when the script ends |
| Hardcoded file paths | `gan_data.csv` assumes the CSV is in the current working directory |
| No containerization | No `Dockerfile` — the environment is not portable |

**What would be needed for production:**

- A `requirements.txt` with pinned versions (e.g., `numpy==1.26.4`)
- Random seed for reproducibility
- Logging with timestamps instead of `print()`
- A `Dockerfile` for containerized execution
- Model artifact saving (e.g., `np.savez()` for weights)
- CI/CD pipeline with health checks

---

## Conclusion

This exercise demonstrates exactly why **MLOps practices matter**. Without dependency management, version pinning, seed control, and proper packaging, even a simple GAN script becomes impossible to reproduce reliably. The gap between "works on my machine" and "runs in production" is enormous.
