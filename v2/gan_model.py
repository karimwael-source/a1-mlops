import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("gan_data.csv")
X = data.values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)  # MinMax scale

latent_dim = 32
data_dim = X.shape[1]

# --- Pure NumPy Neural Network ---
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.2):
    return np.where(x > 0, 1, alpha)

def init_weights(rows, cols):
    return np.random.randn(rows, cols) * 0.02

# Generator weights
g_W1 = init_weights(latent_dim, 64)
g_b1 = np.zeros((1, 64))
g_W2 = init_weights(64, 128)
g_b2 = np.zeros((1, 128))
g_W3 = init_weights(128, data_dim)
g_b3 = np.zeros((1, data_dim))

# Discriminator weights
d_W1 = init_weights(data_dim, 128)
d_b1 = np.zeros((1, 128))
d_W2 = init_weights(128, 64)
d_b2 = np.zeros((1, 64))
d_W3 = init_weights(64, 1)
d_b3 = np.zeros((1, 1))

lr = 0.0002

def generator_forward(z):
    h1 = leaky_relu(z @ g_W1 + g_b1)
    h2 = leaky_relu(h1 @ g_W2 + g_b2)
    out = sigmoid(h2 @ g_W3 + g_b3)
    return h1, h2, out

def discriminator_forward(x):
    h1 = leaky_relu(x @ d_W1 + d_b1)
    h2 = leaky_relu(h1 @ d_W2 + d_b2)
    out = sigmoid(h2 @ d_W3 + d_b3)
    return h1, h2, out

def discriminator_backward(x, h1, h2, out, target):
    global d_W1, d_b1, d_W2, d_b2, d_W3, d_b3
    m = x.shape[0]
    dout = out - target
    dW3 = h2.T @ dout / m
    db3 = np.mean(dout, axis=0, keepdims=True)
    dh2 = dout @ d_W3.T * leaky_relu_deriv(h1 @ d_W2 + d_b2)
    dW2 = h1.T @ dh2 / m
    db2 = np.mean(dh2, axis=0, keepdims=True)
    dh1 = dh2 @ d_W2.T * leaky_relu_deriv(x @ d_W1 + d_b1)
    dW1 = x.T @ dh1 / m
    db1 = np.mean(dh1, axis=0, keepdims=True)
    d_W3 -= lr * dW3
    d_b3 -= lr * db3
    d_W2 -= lr * dW2
    d_b2 -= lr * db2
    d_W1 -= lr * dW1
    d_b1 -= lr * db1

def generator_backward(z, g_h1, g_h2, fake, d_h1, d_h2, d_out):
    global g_W1, g_b1, g_W2, g_b2, g_W3, g_b3
    m = z.shape[0]
    dout = d_out - 1  # generator wants discriminator to output 1
    dd3 = d_h2.T @ dout / m
    ddh2 = dout @ d_W3.T * leaky_relu_deriv(d_h1 @ d_W2 + d_b2)
    ddh1 = ddh2 @ d_W2.T * leaky_relu_deriv(fake @ d_W1 + d_b1)
    dfake = ddh1 @ d_W1.T
    dfake *= fake * (1 - fake)  # sigmoid derivative
    dW3 = g_h2.T @ dfake / m
    db3 = np.mean(dfake, axis=0, keepdims=True)
    dh2 = dfake @ g_W3.T * leaky_relu_deriv(g_h1 @ g_W2 + g_b2)
    dW2 = g_h1.T @ dh2 / m
    db2 = np.mean(dh2, axis=0, keepdims=True)
    dh1 = dh2 @ g_W2.T * leaky_relu_deriv(z @ g_W1 + g_b1)
    dW1 = z.T @ dh1 / m
    db1 = np.mean(dh1, axis=0, keepdims=True)
    g_W3 -= lr * dW3
    g_b3 -= lr * db3
    g_W2 -= lr * dW2
    g_b2 -= lr * db2
    g_W1 -= lr * dW1
    g_b1 -= lr * db1

# Training
epochs = 500
batch_size = 32
d_losses = []
g_losses = []

for epoch in range(epochs):
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_data = X[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_h1, g_h2, fake_data = generator_forward(noise)
    
    # Train discriminator on real
    dh1_r, dh2_r, d_out_r = discriminator_forward(real_data)
    discriminator_backward(real_data, dh1_r, dh2_r, d_out_r, np.ones((batch_size, 1)))
    
    # Train discriminator on fake
    dh1_f, dh2_f, d_out_f = discriminator_forward(fake_data)
    discriminator_backward(fake_data, dh1_f, dh2_f, d_out_f, np.zeros((batch_size, 1)))
    
    d_loss = -np.mean(np.log(d_out_r + 1e-8) + np.log(1 - d_out_f + 1e-8))
    
    # Train generator
    noise2 = np.random.normal(0, 1, (batch_size, latent_dim))
    g_h1, g_h2, fake2 = generator_forward(noise2)
    d_h1, d_h2, d_out = discriminator_forward(fake2)
    g_loss = -np.mean(np.log(d_out + 1e-8))
    generator_backward(noise2, g_h1, g_h2, fake2, d_h1, d_h2, d_out)
    
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    
    if epoch % 100 == 0:
        d_acc_real = np.mean(d_out_r > 0.5) * 100
        d_acc_fake = np.mean(d_out_f < 0.5) * 100
        print(f"Epoch {epoch} | D Loss: {d_loss:.4f} | D Acc Real: {d_acc_real:.1f}% | D Acc Fake: {d_acc_fake:.1f}% | G Loss: {g_loss:.4f}")

# Final evaluation
noise = np.random.normal(0, 1, (len(X), latent_dim))
_, _, generated = generator_forward(noise)
_, _, real_score = discriminator_forward(X)
_, _, fake_score = discriminator_forward(generated)
real_acc = np.mean(real_score > 0.5) * 100
fake_acc = np.mean(fake_score < 0.5) * 100
print(f"\nFinal Discriminator Accuracy on Real: {real_acc:.2f}%")
print(f"Final Discriminator Accuracy on Fake: {fake_acc:.2f}%")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('GAN Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_plot.png')
print("Plot saved to training_plot.png")
