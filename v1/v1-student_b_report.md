# Student B — Reproducibility Report

**Assignment:** MLOps Assignment 1 — Reproducibility Challenge  
**Student A's Code:** `student_a_gan_training.ipynb` (Simple GAN on MNIST)  
**Student B's Attempt:** `student_b_attempt.ipynb`  
**Date:** February 26, 2026

---

## 1. How many commands did I have to run before it worked?

I had to run **3 commands** before the code executed:

1. `pip install numpy` — Not pre-installed; no requirements.txt was provided.
2. `pip install tensorflow` — The main framework; had to guess the correct package name.
3. `python -m notebook` / opened Jupyter — Student A did not specify how to run the notebook.

Student A provided **no `requirements.txt`** and **no documentation**, so I discovered each missing dependency by running the notebook cell-by-cell and fixing `ModuleNotFoundError` errors as they appeared. There was no way to install everything upfront in one step.

---

## 2. What libraries were missing? Did version mismatches cause errors?

**Missing libraries (not documented anywhere):** `numpy`, `tensorflow` (which includes `keras`).

**Version mismatch issues:**

| Library | Student A's Machine | My Machine | Issue |
|---------|-------------------|------------|-------|
| TensorFlow | Unknown (not documented) | 2.20.0 | Could not verify match |
| NumPy | Unknown | 2.4.2 | Could not verify match |
| Python | 3.12.10 | 3.12.10 | Same (by luck) |

- A **Keras deprecation warning** appeared: `"Do not pass an input_shape/input_dim argument to a layer"`. This warning appeared on both machines (visible in the notebook stderr outputs), suggesting Student A also had a newer Keras version, but this warning may become an error in future versions.
- The **Adam optimizer** implementation has changed across TensorFlow versions (learning rate defaults, epsilon values), which can silently produce different training dynamics even with identical code.
- Without pinned versions, running `pip install tensorflow` today installs a potentially different version than what Student A used.

---

## 3. Did the model produce the same result? If not, why?

**No — the results were different.**

| Metric | Student A | Student B (Me) | Difference |
|--------|-----------|---------------|------------|
| Final Disc. Accuracy Score | 0.500258 | 0.500136 | -0.000122 |
| Avg D_loss_real (last 10) | 0.473884 | 0.475358 | +0.001474 |
| Avg D_loss_fake (last 10) | 0.475521 | 0.476977 | +0.001456 |
| Avg G_loss (last 10) | 0.477167 | 0.478618 | +0.001451 |
| **Eval Accuracy** | **0.781250** | **0.825195** | **+0.043945** |
| **Eval F1** | **0.820513** | **0.851205** | **+0.030692** |
| Eval Precision | 0.695652 | 0.740955 | +0.045303 |

**Why the differences?**

1. **Random seeds alone are not enough.** Although Student A set `np.random.seed(123)` and `tf.random.set_seed(123)`, the **weight initialization** happens *before* the seeds are set (at model build time in cells 3-4), so the initial weights differ across runs/machines.
2. **Hardware differences.** Student A ran on Windows (`C:\Users\HD TECH\...`), I ran on Linux. CPU instruction sets (AVX, SSE) and floating-point behavior differ between platforms, causing divergent gradient computations.
3. **TensorFlow version differences.** Different internal implementations of operations (Dense, Adam optimizer) across TF versions lead to numerically different results even with the same seeds.
4. **Evaluation randomness.** The eval section (after training) samples random indices (`np.random.randint`) and noise — the final accuracy depends on *which* samples are evaluated, and this varies each run.

---

## 4. If this had to run on a server at 3:00 AM, would it survive?

**No, it would almost certainly fail.** Here is why:

| Problem | What Would Happen at 3 AM |
|---------|--------------------------|
| **No `requirements.txt`** | Server cannot auto-install dependencies; the job fails immediately on `import tensorflow` |
| **No version pinning** | Even if deps are installed, a new TF version could break the code (e.g., the `input_shape` deprecation warning becoming an error) |
| **Jupyter notebook format** | Notebooks require an interactive kernel; headless servers need `nbconvert` or `papermill` to execute `.ipynb` files — this wasn't documented |
| **MNIST download** | `keras.datasets.mnist.load_data()` downloads from the internet on first run — if the server has no internet access or Google's server is down, it crashes |
| **No error handling** | Any failure (network timeout, OOM, NaN loss) causes an unhandled crash with no alerting |
| **No logging** | Only `print()` statements — output is lost if not captured; no timestamps, no log levels |
| **No model saving** | The trained model exists only in memory; when the script ends, everything is lost |
| **No containerization** | No `Dockerfile`, no `environment.yml` — the environment is not portable or reproducible |
| **No health checks** | No way to verify the job completed successfully or produced valid results |

**What Student A should have provided:**

- `requirements.txt` with pinned versions (e.g., `tensorflow==2.20.0`, `numpy==2.4.2`)
- Random seeds set **before** model creation, not just before training
- A `.py` script (not just `.ipynb`) for headless execution
- Model checkpointing (`model.save()`)
- Proper logging with timestamps
- A `Dockerfile` for containerized, reproducible execution
- Data versioning (e.g., DVC) instead of relying on live downloads

---

## Conclusion

This exercise clearly demonstrates the **reproducibility crisis** in ML. Even with identical code and random seeds, different environments produced different accuracy scores (0.781 vs 0.825). Without proper MLOps practices — dependency management, environment containerization, deterministic execution, and model versioning — ML code is fundamentally unreliable for production use.
