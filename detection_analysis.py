"""
Satellite Detection Performance Analysis Framework
===================================================
Analyzes detection outputs from satellite-based AI models (e.g., Orbitfy)
to evaluate confidence calibration, performance across conditions, and
detection robustness using Monte Carlo sensitivity analysis.

Built with simulated data — framework applies directly to real mission data.

Author: Luka Silbernagel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ============================================================
# SECTION 1: Generate Simulated Satellite Detection Data
# ============================================================
# We simulate what Orbitfy's detection pipeline would output:
# each row is one detection from a satellite pass, with a
# confidence score, ground truth label, and metadata about
# the conditions when the detection was made.
#
# We intentionally build in realistic biases:
#   - Model performs worse at night than during day
#   - Model performs worse over land than ocean (more clutter)
#   - Model is overconfident at high confidence levels
#   - Different target types have different detection difficulty
# These biases are what the analysis framework is designed to find.

def generate_detection_data(n_detections=10000, seed=42):
    """Generate realistic simulated satellite detection data."""
    rng = np.random.default_rng(seed)
    
    # --- Conditions ---
    time_of_day = rng.choice(["day", "night"], size=n_detections, p=[0.6, 0.4])
    terrain = rng.choice(["ocean", "land"], size=n_detections, p=[0.55, 0.45])
    weather = rng.choice(["clear", "cloudy", "overcast"], size=n_detections, p=[0.5, 0.35, 0.15])
    target_class = rng.choice(["vessel", "fire", "vehicle", "aircraft"], 
                               size=n_detections, p=[0.40, 0.25, 0.20, 0.15])
    
    # --- Generate ground truth and confidence scores with realistic biases ---
    # Base true positive rate depends on conditions
    true_positive_prob = np.full(n_detections, 0.85)
    
    # Night degrades performance
    true_positive_prob[time_of_day == "night"] -= 0.12
    
    # Land has more clutter, so more false positives
    true_positive_prob[terrain == "land"] -= 0.08
    
    # Overcast weather degrades optical detection
    true_positive_prob[weather == "overcast"] -= 0.10
    true_positive_prob[weather == "cloudy"] -= 0.04
    
    # Some targets are harder to detect
    true_positive_prob[target_class == "vehicle"] -= 0.06
    true_positive_prob[target_class == "aircraft"] -= 0.03
    
    # Clip to valid probability range
    true_positive_prob = np.clip(true_positive_prob, 0.15, 0.98)
    
    # Ground truth: is this detection actually a real target?
    ground_truth = rng.random(n_detections) < true_positive_prob
    
    # Generate confidence scores
    # True positives get higher confidence, false positives get lower
    confidence = np.zeros(n_detections)
    
    tp_mask = ground_truth == True
    fp_mask = ground_truth == False
    
    # True positives: confidence centered around 0.80
    confidence[tp_mask] = rng.beta(8, 2, size=tp_mask.sum())
    
    # False positives: confidence centered around 0.45
    confidence[fp_mask] = rng.beta(4, 5, size=fp_mask.sum())
    
    # Add systematic overconfidence bias at high confidence levels
    # (common in real ML models — they say 95% but are only right 88%)
    high_conf_mask = confidence > 0.8
    confidence[high_conf_mask] = np.clip(
        confidence[high_conf_mask] + rng.normal(0.05, 0.02, size=high_conf_mask.sum()),
        0, 1
    )
    
    # --- Build dataframe ---
    data = pd.DataFrame({
        "detection_id": np.arange(n_detections),
        "confidence": np.round(confidence, 4),
        "ground_truth": ground_truth.astype(int),
        "target_class": target_class,
        "time_of_day": time_of_day,
        "terrain": terrain,
        "weather": weather,
        "latitude": np.round(rng.uniform(-60, 60, n_detections), 4),
        "longitude": np.round(rng.uniform(-180, 180, n_detections), 4),
    })
    
    return data


# ============================================================
# SECTION 2: Confidence Calibration Analysis
# ============================================================
# Key question: when the model says "90% confident," is it
# actually right 90% of the time?
#
# A well-calibrated model's confidence scores match reality.
# An overconfident model says 90% but is only right 75%.
# This matters because operators make decisions based on these
# scores — if they can't trust them, the system is unreliable.
#
# Method: bin detections by confidence level, compute the
# actual accuracy in each bin, and compare.

def calibration_analysis(data, n_bins=10):
    """Analyze whether model confidence scores match actual accuracy."""
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    actual_accuracy = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (data["confidence"] >= bin_edges[i]) & (data["confidence"] < bin_edges[i + 1])
        bin_data = data[mask]
        
        if len(bin_data) > 0:
            accuracy = bin_data["ground_truth"].mean()
            actual_accuracy.append(accuracy)
            bin_counts.append(len(bin_data))
        else:
            actual_accuracy.append(np.nan)
            bin_counts.append(0)
    
    return bin_centers, np.array(actual_accuracy), np.array(bin_counts)


def plot_calibration(bin_centers, actual_accuracy, bin_counts, output_dir):
    """Plot calibration curve: predicted confidence vs actual accuracy."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1])
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.plot(bin_centers, actual_accuracy, "o-", color="#2563eb", linewidth=2,
             markersize=8, label="Model calibration")
    
    # Shade the gap between perfect and actual
    ax1.fill_between(bin_centers, bin_centers, actual_accuracy, 
                      alpha=0.15, color="#ef4444")
    
    ax1.set_xlabel("Predicted Confidence", fontsize=12)
    ax1.set_ylabel("Actual Accuracy (fraction correct)", fontsize=12)
    ax1.set_title("Confidence Calibration Analysis", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Histogram of confidence distribution
    ax2.bar(bin_centers, bin_counts, width=0.08, color="#2563eb", alpha=0.7)
    ax2.set_xlabel("Confidence Score", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Detection Volume by Confidence Level", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Compute calibration error (ECE - Expected Calibration Error)
    weights = bin_counts / bin_counts.sum()
    valid = ~np.isnan(actual_accuracy)
    ece = np.sum(weights[valid] * np.abs(bin_centers[valid] - actual_accuracy[valid]))
    
    return ece


# ============================================================
# SECTION 3: Performance Breakdown by Condition
# ============================================================
# The model doesn't perform equally in all conditions.
# We want to know: where does it work well, and where does
# it fail? This is critical for mission planning — if the
# model is unreliable at night, operators need to know that.
#
# Metrics:
#   Precision = of all detections flagged, what % were real
#   Recall    = of all real targets, what % were detected
#   F1 Score  = harmonic mean of precision and recall

def compute_metrics(data, threshold=0.5):
    """Compute precision, recall, and F1 for a given confidence threshold."""
    
    predicted_positive = data["confidence"] >= threshold
    actual_positive = data["ground_truth"] == 1
    
    tp = (predicted_positive & actual_positive).sum()
    fp = (predicted_positive & ~actual_positive).sum()
    fn = (~predicted_positive & actual_positive).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, 
            "true_positives": tp, "false_positives": fp, "missed": fn}


def performance_breakdown(data, threshold=0.5):
    """Break down performance by every condition variable."""
    
    results = {}
    
    for condition in ["time_of_day", "terrain", "weather", "target_class"]:
        condition_results = {}
        for value in data[condition].unique():
            subset = data[data[condition] == value]
            metrics = compute_metrics(subset, threshold)
            metrics["n_detections"] = len(subset)
            condition_results[value] = metrics
        results[condition] = condition_results
    
    return results


def plot_performance_breakdown(results, output_dir):
    """Plot performance metrics broken down by each condition."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    titles = {
        "time_of_day": "Time of Day",
        "terrain": "Terrain Type", 
        "weather": "Weather Conditions",
        "target_class": "Target Class"
    }
    
    for idx, (condition, condition_results) in enumerate(results.items()):
        ax = axes[idx // 2][idx % 2]
        
        categories = list(condition_results.keys())
        precision = [condition_results[c]["precision"] for c in categories]
        recall = [condition_results[c]["recall"] for c in categories]
        f1 = [condition_results[c]["f1"] for c in categories]
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(x - width, precision, width, label="Precision", color="#2563eb", alpha=0.8)
        ax.bar(x, recall, width, label="Recall", color="#16a34a", alpha=0.8)
        ax.bar(x + width, f1, width, label="F1 Score", color="#ea580c", alpha=0.8)
        
        ax.set_xlabel(titles[condition], fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Performance by {titles[condition]}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("Detection Performance Across Conditions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# SECTION 4: Monte Carlo Sensitivity Analysis
# ============================================================
# Key question: how robust are the detection decisions?
#
# When the model says 87% confidence on a detection, how
# stable is that number? If you slightly perturb the input
# (add noise, shift the image), does it stay at 87% or
# swing wildly between 50% and 99%?
#
# Detections with stable confidence are trustworthy.
# Detections with volatile confidence are fragile — the model
# is uncertain even if any single run looks confident.
#
# This is the same Monte Carlo philosophy as the SSP asteroid
# work: perturb the inputs, see how the outputs change,
# and use the distribution to quantify true uncertainty.

def sensitivity_analysis(data, n_simulations=1000, noise_std=0.05, seed=42):
    """
    Monte Carlo sensitivity analysis on detection confidence scores.
    
    Simulates what would happen if the input data had small perturbations
    (noise, atmospheric variation, sensor drift) by adding Gaussian noise
    to confidence scores and checking how often the detection decision
    (accept/reject at threshold) changes.
    
    Returns a "robustness score" for each detection:
      - 1.0 = decision never changed across all simulations (very robust)
      - 0.5 = decision changed half the time (very fragile)
    """
    rng = np.random.default_rng(seed)
    
    base_confidence = data["confidence"].values
    threshold = 0.5
    base_decision = base_confidence >= threshold
    
    # Run Monte Carlo: add noise to confidence scores and check stability
    decision_agreement = np.zeros(len(data))
    perturbed_confidences = np.zeros((n_simulations, len(data)))
    
    for i in range(n_simulations):
        noise = rng.normal(0, noise_std, size=len(data))
        perturbed = np.clip(base_confidence + noise, 0, 1)
        perturbed_decision = perturbed >= threshold
        decision_agreement += (perturbed_decision == base_decision).astype(float)
        perturbed_confidences[i] = perturbed
    
    robustness_score = decision_agreement / n_simulations
    confidence_std = perturbed_confidences.std(axis=0)
    
    return robustness_score, confidence_std


def plot_sensitivity(data, robustness_score, confidence_std, output_dir):
    """Plot Monte Carlo sensitivity analysis results."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Robustness score distribution
    ax1 = axes[0]
    ax1.hist(robustness_score, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    ax1.axvline(x=0.9, color="#ef4444", linestyle="--", linewidth=2, label="Robustness threshold (0.9)")
    fragile_pct = (robustness_score < 0.9).mean() * 100
    ax1.set_xlabel("Robustness Score", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Detection Robustness Distribution\n{fragile_pct:.1f}% of detections are fragile (<0.9)", 
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Robustness vs confidence — where are the fragile detections?
    ax2 = axes[1]
    scatter = ax2.scatter(data["confidence"], robustness_score, 
                          c=data["ground_truth"], cmap="RdYlGn",
                          alpha=0.3, s=10)
    ax2.axhline(y=0.9, color="#ef4444", linestyle="--", linewidth=1.5)
    ax2.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Original Confidence Score", fontsize=12)
    ax2.set_ylabel("Robustness Score", fontsize=12)
    ax2.set_title("Robustness vs Confidence\n(green = true positive, red = false positive)", 
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confidence volatility (std) by confidence bin
    ax3 = axes[2]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_stds = []
    for i in range(len(bins) - 1):
        mask = (data["confidence"] >= bins[i]) & (data["confidence"] < bins[i + 1])
        if mask.sum() > 0:
            bin_stds.append(confidence_std[mask].mean())
        else:
            bin_stds.append(0)
    
    ax3.bar(bin_centers, bin_stds, width=0.08, color="#f59e0b", alpha=0.8, edgecolor="white")
    ax3.set_xlabel("Confidence Score Bin", fontsize=12)
    ax3.set_ylabel("Mean Confidence Volatility (σ)", fontsize=12)
    ax3.set_title("Confidence Volatility by Score Range", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle("Monte Carlo Sensitivity Analysis (1,000 simulations, σ=0.05 noise)", 
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return fragile_pct


# ============================================================
# SECTION 5: Summary Report
# ============================================================

def print_report(data, ece, breakdown, robustness_score, fragile_pct):
    """Print a summary of all findings."""
    
    print("=" * 65)
    print("  SATELLITE DETECTION PERFORMANCE ANALYSIS REPORT")
    print("=" * 65)
    
    print(f"\nDataset: {len(data)} detections")
    print(f"True positive rate: {data['ground_truth'].mean():.1%}")
    print(f"Mean confidence score: {data['confidence'].mean():.3f}")
    
    # Calibration
    print(f"\n--- CALIBRATION ---")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    if ece > 0.05:
        print("⚠  Model is MISCALIBRATED — confidence scores do not match")
        print("   actual accuracy. High-confidence detections are less")
        print("   reliable than their scores suggest.")
    else:
        print("✓  Model is well-calibrated.")
    
    # Performance breakdown
    print(f"\n--- PERFORMANCE BY CONDITION (threshold=0.5) ---")
    for condition, condition_results in breakdown.items():
        print(f"\n  {condition.upper().replace('_', ' ')}:")
        for value, metrics in condition_results.items():
            print(f"    {value:12s}  P={metrics['precision']:.3f}  "
                  f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  "
                  f"(n={metrics['n_detections']})")
    
    # Worst conditions
    print(f"\n--- KEY FINDINGS ---")
    all_f1s = {}
    for condition, condition_results in breakdown.items():
        for value, metrics in condition_results.items():
            all_f1s[f"{condition}={value}"] = metrics["f1"]
    
    worst = sorted(all_f1s.items(), key=lambda x: x[1])[:3]
    best = sorted(all_f1s.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("  Weakest conditions:")
    for name, f1 in worst:
        print(f"    • {name}: F1={f1:.3f}")
    
    print("  Strongest conditions:")
    for name, f1 in best:
        print(f"    • {name}: F1={f1:.3f}")
    
    # Sensitivity
    print(f"\n--- ROBUSTNESS (Monte Carlo, n=1000) ---")
    print(f"  Fragile detections (<0.9 robustness): {fragile_pct:.1f}%")
    print(f"  Mean robustness score: {robustness_score.mean():.3f}")
    
    # Fragile detections near the decision boundary
    near_boundary = (data["confidence"] > 0.4) & (data["confidence"] < 0.6)
    if near_boundary.sum() > 0:
        boundary_fragile = (robustness_score[near_boundary] < 0.9).mean() * 100
        print(f"  Fragile detections near boundary (0.4-0.6): {boundary_fragile:.1f}%")
        print("  → Detections near the decision threshold are most vulnerable")
        print("    to input perturbations. Consider a two-threshold system:")
        print("    accept above 0.7, reject below 0.3, flag middle for review.")
    
    print(f"\n{'=' * 65}")
    print("  Analysis complete. Plots saved to output/")
    print(f"{'=' * 65}")


# ============================================================
# MAIN — Run the full analysis pipeline
# ============================================================

if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating simulated detection data...")
    data = generate_detection_data(n_detections=10000)
    data.to_csv(output_dir / "simulated_detections.csv", index=False)
    print(f"  Generated {len(data)} detections → output/simulated_detections.csv")
    
    print("\nRunning calibration analysis...")
    bin_centers, actual_accuracy, bin_counts = calibration_analysis(data)
    ece = plot_calibration(bin_centers, actual_accuracy, bin_counts, output_dir)
    print(f"  ECE = {ece:.4f} → output/calibration_curve.png")
    
    print("\nRunning performance breakdown...")
    breakdown = performance_breakdown(data, threshold=0.5)
    plot_performance_breakdown(breakdown, output_dir)
    print(f"  → output/performance_breakdown.png")
    
    print("\nRunning Monte Carlo sensitivity analysis (1,000 simulations)...")
    robustness_score, confidence_std = sensitivity_analysis(data)
    fragile_pct = plot_sensitivity(data, robustness_score, confidence_std, output_dir)
    print(f"  {fragile_pct:.1f}% fragile detections → output/sensitivity_analysis.png")
    
    print("\n")
    print_report(data, ece, breakdown, robustness_score, fragile_pct)
