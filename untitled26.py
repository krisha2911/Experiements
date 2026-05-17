"""
Experiment No. 1 - Hidden Markov Model for Weather Prediction
Aim: To design and implement a HMM for weather prediction using observable daily activities.
"""

import numpy as np

# ─────────────────────────────────────────────
# 1. MODEL DEFINITION
# ─────────────────────────────────────────────

# Hidden states (weather)
states = ["Sunny", "Rainy"]
n_states = len(states)

# Observable activities
observations = ["Walk", "Shop", "Clean"]
n_obs = len(observations)

# Initial state probabilities  π
pi = np.array([0.6, 0.4])   # P(Sunny=0.6), P(Rainy=0.4)

# State Transition Matrix  A
# A[i][j] = P(next state = j | current state = i)
A = np.array([
    [0.7, 0.3],   # Sunny → [Sunny, Rainy]
    [0.4, 0.6],   # Rainy → [Sunny, Rainy]
])

# Emission (Observation) Probability Matrix  B
# B[i][k] = P(observation = k | state = i)
B = np.array([
    [0.6, 0.3, 0.1],   # Sunny → [Walk, Shop, Clean]
    [0.1, 0.4, 0.5],   # Rainy → [Walk, Shop, Clean]
])

obs_index = {o: i for i, o in enumerate(observations)}
state_index = {s: i for i, s in enumerate(states)}


# ─────────────────────────────────────────────
# 2. VITERBI ALGORITHM
# ─────────────────────────────────────────────

def viterbi(obs_seq):
    """
    Find the most likely hidden state sequence for a given observation sequence.
    Returns the best path and its probability.
    """
    T = len(obs_seq)
    dp   = np.zeros((n_states, T))   # dp[s][t]  = max probability
    psi  = np.zeros((n_states, T), dtype=int)  # backpointer

    # Initialisation (t = 0)
    o0 = obs_index[obs_seq[0]]
    dp[:, 0] = pi * B[:, o0]

    # Recursion
    for t in range(1, T):
        ot = obs_index[obs_seq[t]]
        for s in range(n_states):
            trans_prob = dp[:, t-1] * A[:, s]
            psi[s, t]  = np.argmax(trans_prob)
            dp[s, t]   = np.max(trans_prob) * B[s, ot]

    # Backtracking
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(dp[:, T-1])
    for t in range(T-2, -1, -1):
        best_path[t] = psi[best_path[t+1], t+1]

    best_prob = np.max(dp[:, T-1])
    return [states[s] for s in best_path], best_prob


# ─────────────────────────────────────────────
# 3. EVALUATION METRICS
# ─────────────────────────────────────────────

def evaluate(predicted, actual):
    correct = sum(p == a for p, a in zip(predicted, actual))
    accuracy = correct / len(actual) * 100
    return accuracy


# ─────────────────────────────────────────────
# 4. RESULTS & VISUALISATION (text-based)
# ─────────────────────────────────────────────

def print_matrices():
    print("\n── Model Parameters ──────────────────────────")
    print(f"States      : {states}")
    print(f"Observations: {observations}")
    print(f"\nInitial Probabilities (π):\n  {dict(zip(states, pi))}")

    print("\nTransition Matrix (A):")
    header = f"{'':>8}" + "".join(f"{s:>8}" for s in states)
    print(header)
    for i, row in enumerate(A):
        print(f"  {states[i]:>6}" + "".join(f"{v:>8.2f}" for v in row))

    print("\nEmission Matrix (B):")
    header = f"{'':>8}" + "".join(f"{o:>8}" for o in observations)
    print(header)
    for i, row in enumerate(B):
        print(f"  {states[i]:>6}" + "".join(f"{v:>8.2f}" for v in row))


def visualise(obs_seq, predicted):
    print("\n── Prediction Results ────────────────────────")
    print(f"  {'Day':<5} {'Observed Activity':<20} {'Predicted Weather'}")
    print(f"  {'-'*5} {'-'*20} {'-'*18}")
    for day, (obs, pred) in enumerate(zip(obs_seq, predicted), start=1):
        icon = "☀️ " if pred == "Sunny" else "🌧️ "
        print(f"  {day:<5} {obs:<20} {icon}{pred}")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("   HMM Weather Prediction — Experiment No. 1")
    print("=" * 52)

    print_matrices()

    # Observation sequence (daily activities)
    obs_sequence = ["Walk", "Shop", "Clean", "Clean", "Walk", "Shop"]

    # Known weather for evaluation
    actual_weather = ["Sunny", "Sunny", "Rainy", "Rainy", "Sunny", "Sunny"]

    print(f"\n── Input Observations ────────────────────────")
    print(f"  {obs_sequence}")

    predicted_weather, best_prob = viterbi(obs_sequence)

    visualise(obs_sequence, predicted_weather)

    print(f"\n── Viterbi Probability ───────────────────────")
    print(f"  Best path probability: {best_prob:.6f}")

    accuracy = evaluate(predicted_weather, actual_weather)
    print(f"\n── Evaluation Metric ─────────────────────────")
    print(f"  Actual   : {actual_weather}")
    print(f"  Predicted: {predicted_weather}")
    print(f"  Accuracy : {accuracy:.1f}%")
    print("=" * 52)




EXP 2



"""
Experiment No. 2 - Bayesian Network for Student Exam Pass Prediction
Aim: To design and implement a Bayesian Network for predicting whether a student
     will pass a final exam based on study habits, class attendance, and
     internal assessment performance.
"""
!pip install pgmpy
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

# ─────────────────────────────────────────────
# 1. NETWORK STRUCTURE
# ─────────────────────────────────────────────
# Nodes:
#   S  = Study Habits      (0 = Poor,  1 = Good)
#   A  = Class Attendance  (0 = Low,   1 = High)
#   I  = Internal Score    (0 = Fail,  1 = Pass)  ← depends on S, A
#   P  = Pass Final Exam   (0 = Fail,  1 = Pass)  ← depends on S, A, I

model = BayesianNetwork([
    ("Study",      "Internal"),
    ("Attendance", "Internal"),
    ("Study",      "Pass"),
    ("Attendance", "Pass"),
    ("Internal",   "Pass"),
])

# ─────────────────────────────────────────────
# 2. CONDITIONAL PROBABILITY DISTRIBUTIONS
# ─────────────────────────────────────────────

# P(Study)
cpd_study = TabularCPD(
    variable="Study", variable_card=2,
    values=[[0.4],   # Poor
            [0.6]],  # Good
)

# P(Attendance)
cpd_attend = TabularCPD(
    variable="Attendance", variable_card=2,
    values=[[0.35],  # Low
            [0.65]], # High
)

# P(Internal | Study, Attendance)
#             S=Poor,A=Low  S=Poor,A=High  S=Good,A=Low  S=Good,A=High
cpd_internal = TabularCPD(
    variable="Internal", variable_card=2,
    values=[
        [0.85, 0.60, 0.50, 0.20],   # Fail
        [0.15, 0.40, 0.50, 0.80],   # Pass
    ],
    evidence=["Study", "Attendance"],
    evidence_card=[2, 2],
)

# P(Pass | Study, Attendance, Internal)
#   evidence order: Study(2) x Attendance(2) x Internal(2) = 8 columns
#   col order: S=0,A=0,I=0 | S=0,A=0,I=1 | S=0,A=1,I=0 | S=0,A=1,I=1
#              S=1,A=0,I=0 | S=1,A=0,I=1 | S=1,A=1,I=0 | S=1,A=1,I=1
cpd_pass = TabularCPD(
    variable="Pass", variable_card=2,
    values=[
        [0.95, 0.70, 0.80, 0.50, 0.75, 0.40, 0.55, 0.10],  # Fail
        [0.05, 0.30, 0.20, 0.50, 0.25, 0.60, 0.45, 0.90],  # Pass
    ],
    evidence=["Study", "Attendance", "Internal"],
    evidence_card=[2, 2, 2],
)

# ─────────────────────────────────────────────
# 3. ASSEMBLE & VALIDATE MODEL
# ─────────────────────────────────────────────
model.add_cpds(cpd_study, cpd_attend, cpd_internal, cpd_pass)
assert model.check_model(), "Model is invalid!"

# ─────────────────────────────────────────────
# 4. INFERENCE — Variable Elimination
# ─────────────────────────────────────────────
infer = VariableElimination(model)

def predict(study=None, attendance=None, internal=None):
    """Query P(Pass | given evidence). Pass None to marginalise."""
    evidence = {}
    if study      is not None: evidence["Study"]      = study
    if attendance is not None: evidence["Attendance"] = attendance
    if internal   is not None: evidence["Internal"]   = internal
    result = infer.query(variables=["Pass"], evidence=evidence, show_progress=False)
    return result.values[1]   # P(Pass=1)

# ─────────────────────────────────────────────
# 5. PRINT HELPERS
# ─────────────────────────────────────────────
def print_cpds():
    labels = {"Study": ["Poor","Good"], "Attendance": ["Low","High"],
              "Internal": ["Fail","Pass"], "Pass": ["Fail","Pass"]}
    print("\n── CPD: Study ────────────────────────────────")
    print(f"  P(Poor) = 0.40   P(Good) = 0.60")

    print("\n── CPD: Attendance ───────────────────────────")
    print(f"  P(Low)  = 0.35   P(High) = 0.65")

    print("\n── CPD: Internal | Study, Attendance ─────────")
    print(f"  {'Study':>8} {'Attend':>8} {'P(Fail)':>9} {'P(Pass)':>9}")
    print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
    for s, sl in enumerate(["Poor","Good"]):
        for a, al in enumerate(["Low","High"]):
            pf = float(cpd_internal.values[0, s, a])
            pp = float(cpd_internal.values[1, s, a])
            print(f"  {sl:>8} {al:>8} {pf:>9.2f} {pp:>9.2f}")

def print_scenarios():
    print("\n── Inference Scenarios ───────────────────────")
    scenarios = [
        ("Good study, High attend, Pass internal", 1, 1, 1),
        ("Poor study, Low attend,  Fail internal", 0, 0, 0),
        ("Good study, Low attend,  Fail internal", 1, 0, 0),
        ("Poor study, High attend, Pass internal", 0, 1, 1),
        ("No evidence (prior)",                   None, None, None),
    ]
    print(f"  {'Scenario':<42} {'P(Pass Exam)':>12}")
    print(f"  {'-'*42} {'-'*12}")
    for label, s, a, i in scenarios:
        p = predict(s, a, i)
        bar = "█" * int(p * 20)
        print(f"  {label:<42} {p:>10.4f}  {bar}")

def print_sensitivity():
    print("\n── Sensitivity: Effect of Each Factor ────────")
    cases = [
        ("Only Good Study",      dict(study=1)),
        ("Only Poor Study",      dict(study=0)),
        ("Only High Attendance", dict(attendance=1)),
        ("Only Low  Attendance", dict(attendance=0)),
        ("Only Pass Internal",   dict(internal=1)),
        ("Only Fail Internal",   dict(internal=0)),
    ]
    print(f"  {'Condition':<25} {'P(Pass Exam)':>12}")
    print(f"  {'-'*25} {'-'*12}")
    for label, ev in cases:
        p = predict(**ev)
        print(f"  {label:<25} {p:>12.4f}")

# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 56)
    print("  Bayesian Network — Student Exam Prediction (Exp 2)")
    print("=" * 56)

    print("\n── Network Structure ─────────────────────────")
    print("  Study ──┐")
    print("           ├──► Internal ──► Pass")
    print("  Attend ──┘         ↑")
    print("                Study, Attend also → Pass directly")

    print_cpds()
    print_scenarios()
    print_sensitivity()

    print("\n── Best-case vs Worst-case ───────────────────")
    best  = predict(study=1, attendance=1, internal=1)
    worst = predict(study=0, attendance=0, internal=0)
    print(f"  Best  (Good study, High attend, Pass internal): {best:.4f}")
    print(f"  Worst (Poor study, Low  attend, Fail internal): {worst:.4f}")
    print("=" * 56)





"""
Experiment No. 3 - Gaussian Mixture Model (GMM) for Outcome Prediction
Aim: To design and implement a GMM for outcome prediction on a real-world dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS REAL-WORLD DATASET
# ─────────────────────────────────────────────
iris = load_iris()
X = iris.data          # 150 samples, 4 features
y = iris.target        # true labels (used only for evaluation)
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=" * 55)
print("   Gaussian Mixture Model — Experiment No. 3")
print("=" * 55)
print(f"\nDataset      : Iris (real-world)")
print(f"Samples      : {X.shape[0]}")
print(f"Features     : {X.shape[1]} ({', '.join(iris.feature_names)})")
print(f"True Classes : {list(target_names)}")

# ─────────────────────────────────────────────
# 2. FIT GMM
# ─────────────────────────────────────────────
n_components = 3   # one Gaussian per flower species

gmm = GaussianMixture(
    n_components=n_components,
    covariance_type="full",   # each component has its own covariance matrix
    max_iter=200,
    random_state=42
)
gmm.fit(X_scaled)

print(f"\n── GMM Parameters ────────────────────────────")
print(f"  Components     : {n_components}")
print(f"  Covariance type: full")
print(f"  Converged      : {gmm.converged_}")
print(f"  Log-Likelihood : {gmm.lower_bound_:.4f}")
print(f"  AIC            : {gmm.aic(X_scaled):.2f}")
print(f"  BIC            : {gmm.bic(X_scaled):.2f}")

print(f"\n  Mixing weights (π): {np.round(gmm.weights_, 4)}")

# ─────────────────────────────────────────────
# 3. PREDICT CLUSTER LABELS
# ─────────────────────────────────────────────
gmm_labels = gmm.predict(X_scaled)
probs       = gmm.predict_proba(X_scaled)   # soft assignments

# Align GMM cluster IDs to true class IDs (greedy matching)
from scipy.stats import mode
mapping = {}
for cluster in range(n_components):
    mask = gmm_labels == cluster
    true_majority = mode(y[mask], keepdims=True).mode[0]
    mapping[cluster] = true_majority

aligned = np.array([mapping[c] for c in gmm_labels])
acc = accuracy_score(y, aligned)

# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────
print(f"\n── Evaluation Metrics ────────────────────────")
print(f"  Accuracy : {acc*100:.2f}%\n")
print("  Classification Report:")
print(classification_report(y, aligned, target_names=target_names))

print("  Confusion Matrix:")
cm = confusion_matrix(y, aligned)
print(f"  {'':>12}", end="")
for name in target_names:
    print(f"{name:>12}", end="")
print()
for i, row in enumerate(cm):
    print(f"  {target_names[i]:>12}", end="")
    for val in row:
        print(f"{val:>12}", end="")
    print()

# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Gaussian Mixture Model — Iris Dataset", fontsize=14, fontweight="bold")

colors_true = ["#e74c3c", "#2ecc71", "#3498db"]
colors_pred = ["#f39c12", "#9b59b6", "#1abc9c"]

# Plot 1 — True labels
ax = axes[0]
for i, name in enumerate(target_names):
    mask = y == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colors_true[i], label=name, s=40, alpha=0.8)
ax.set_title("True Labels")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 2 — GMM predicted clusters
ax = axes[1]
for i in range(n_components):
    mask = aligned == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colors_true[i], label=target_names[i], s=40, alpha=0.8)
ax.set_title("GMM Predicted Clusters")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 3 — Soft assignment confidence (max probability)
ax = axes[2]
confidence = probs.max(axis=1)
sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=confidence,
                cmap="RdYlGn", s=40, alpha=0.9)
plt.colorbar(sc, ax=ax, label="Max P(cluster|x)")
ax.set_title("Soft Assignment Confidence")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gmm_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Plot saved as gmm_results.png")

# ─────────────────────────────────────────────
# 6. MODEL SELECTION — BIC across K values
# ─────────────────────────────────────────────
print("\n── Model Selection (BIC vs n_components) ────")
bic_scores, aic_scores = [], []
K_range = range(1, 8)
for k in K_range:
    g = GaussianMixture(n_components=k, covariance_type="full",
                        random_state=42, max_iter=200)
    g.fit(X_scaled)
    bic_scores.append(g.bic(X_scaled))
    aic_scores.append(g.aic(X_scaled))

best_k = K_range[np.argmin(bic_scores)]
print(f"  {'K':>4} {'BIC':>10} {'AIC':>10}")
print(f"  {'-'*4} {'-'*10} {'-'*10}")
for k, b, a in zip(K_range, bic_scores, aic_scores):
    marker = " ← best" if k == best_k else ""
    print(f"  {k:>4} {b:>10.2f} {a:>10.2f}{marker}")

plt.figure(figsize=(6, 4))
plt.plot(K_range, bic_scores, marker="o", label="BIC", color="#e74c3c")
plt.plot(K_range, aic_scores, marker="s", label="AIC", color="#3498db")
plt.axvline(best_k, linestyle="--", color="gray", alpha=0.7, label=f"Best K={best_k}")
plt.xlabel("Number of Components (K)")
plt.ylabel("Score")
plt.title("GMM Model Selection: BIC & AIC")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gmm_model_selection.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Model selection plot saved.")
print("=" * 55)







"""
Experiment No. 4 - Generative Multi-Layer Network Model
Aim: To build and train a Generative Multi-Layer Network (MLP) that maps a
     low-dimensional latent space to a target data distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# ─────────────────────────────────────────────
# 1. SETUP & TARGET DATA
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# Use Iris as real-world target distribution
iris = load_iris()
X_raw = iris.data.astype(np.float32)
scaler = StandardScaler()
X_real = scaler.fit_transform(X_raw)

X_tensor = torch.tensor(X_real, dtype=torch.float32)

LATENT_DIM  = 2     # low-dimensional input (latent space)
OUTPUT_DIM  = 4     # target data dimensions (Iris has 4 features)
HIDDEN_DIM  = 64
EPOCHS      = 2000
LR          = 1e-3
BATCH_SIZE  = 32

print("=" * 55)
print("  Generative Multi-Layer Network — Experiment 4")
print("=" * 55)
print(f"\n  Dataset    : Iris (real-world, {X_real.shape[0]} samples)")
print(f"  Latent dim : {LATENT_DIM}  →  Output dim : {OUTPUT_DIM}")

# ─────────────────────────────────────────────
# 2. GENERATOR NETWORK (MLP)
# ─────────────────────────────────────────────
class GeneratorMLP(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)

generator = GeneratorMLP(LATENT_DIM, HIDDEN_DIM, OUTPUT_DIM)

total_params = sum(p.numel() for p in generator.parameters())
print(f"\n── Network Architecture ──────────────────────")
print(generator.net)
print(f"\n  Total trainable parameters: {total_params}")

# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────
optimizer = optim.Adam(generator.parameters(), lr=LR)
criterion = nn.MSELoss()

loss_history = []

print(f"\n── Training ──────────────────────────────────")
for epoch in range(1, EPOCHS + 1):
    # Sample random batch from real data
    idx = torch.randint(0, len(X_tensor), (BATCH_SIZE,))
    real_batch = X_tensor[idx]

    # Sample latent vectors z ~ N(0, I)
    z = torch.randn(BATCH_SIZE, LATENT_DIM)

    # Generate fake samples
    fake_batch = generator(z)

    # Loss: match generated distribution to real data statistics
    # Mean and covariance matching (moment matching loss)
    real_mean = real_batch.mean(dim=0)
    fake_mean = fake_batch.mean(dim=0)
    real_std  = real_batch.std(dim=0)
    fake_std  = fake_batch.std(dim=0)

    loss_mean = criterion(fake_mean, real_mean)
    loss_std  = criterion(fake_std,  real_std)
    loss      = loss_mean + loss_std

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 400 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:>4}/{EPOCHS}]  Loss: {loss.item():.6f}"
              f"  (mean: {loss_mean.item():.4f}, std: {loss_std.item():.4f})")

# ─────────────────────────────────────────────
# 4. GENERATE SAMPLES & EVALUATE
# ─────────────────────────────────────────────
generator.eval()
with torch.no_grad():
    z_eval   = torch.randn(500, LATENT_DIM)
    gen_data = generator(z_eval).numpy()

real_mean_np = X_real.mean(axis=0)
gen_mean_np  = gen_data.mean(axis=0)
real_std_np  = X_real.std(axis=0)
gen_std_np   = gen_data.std(axis=0)

feature_names = [f.replace(" (cm)", "") for f in iris.feature_names]

print(f"\n── Distribution Comparison ───────────────────")
print(f"  {'Feature':<20} {'Real μ':>8} {'Gen μ':>8} {'Real σ':>8} {'Gen σ':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for i, feat in enumerate(feature_names):
    print(f"  {feat:<20} {real_mean_np[i]:>8.4f} {gen_mean_np[i]:>8.4f}"
          f" {real_std_np[i]:>8.4f} {gen_std_np[i]:>8.4f}")

mean_err = np.abs(real_mean_np - gen_mean_np).mean()
std_err  = np.abs(real_std_np  - gen_std_np).mean()
print(f"\n  Mean absolute error (means) : {mean_err:.4f}")
print(f"  Mean absolute error (stds)  : {std_err:.4f}")

# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Generative MLP — Latent Space to Data Distribution", fontsize=14, fontweight="bold")

# Plot 1 — Training loss curve
ax = axes[0, 0]
ax.plot(loss_history, color="#e74c3c", alpha=0.8, linewidth=1)
ax.set_title("Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3)

# Plots 2–5 — Feature distributions (real vs generated)
for i, feat in enumerate(feature_names):
    r, c = divmod(i + 1, 3)
    ax = axes[r, c]
    ax.hist(X_real[:, i],   bins=25, alpha=0.6, color="#3498db", label="Real",      density=True)
    ax.hist(gen_data[:, i], bins=25, alpha=0.6, color="#e74c3c", label="Generated", density=True)
    ax.set_title(feat); ax.set_xlabel("Value (standardised)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 6 — Latent space scatter (colour by generated feature 0)
ax = axes[1, 2]
z_plot = z_eval.numpy()
sc = ax.scatter(z_plot[:, 0], z_plot[:, 1], c=gen_data[:, 0],
                cmap="viridis", s=15, alpha=0.7)
plt.colorbar(sc, ax=ax, label=feature_names[0])
ax.set_title("Latent Space (z₁ vs z₂)"); ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("generative_mlp_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Plot saved as generative_mlp_results.png")
print("=" * 55)
