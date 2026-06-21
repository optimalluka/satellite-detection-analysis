# Satellite Detection Performance Analysis

A Python framework for evaluating AI-based satellite detection systems. Analyzes confidence calibration, performance degradation across operating conditions, and detection robustness using Monte Carlo sensitivity analysis.

Built as a generalizable tool — uses simulated data but the framework applies directly to real mission outputs from systems like [Orbitfy](https://www.littleplace.com).

## What It Does

**Confidence Calibration Analysis**  
Checks whether the model's confidence scores match actual accuracy. When the model says "90% confident," is it right 90% of the time? Miscalibrated models give operators false confidence in unreliable detections.

**Performance Breakdown by Condition**  
Measures precision, recall, and F1 score across every operating variable — time of day, terrain type, weather, and target class. Identifies the specific conditions where the model underperforms, which is critical for mission planning and operator trust.

**Monte Carlo Sensitivity Analysis**  
Perturbs detection confidence scores across 1,000 simulations to measure how robust each detection decision is to input noise. Produces a per-detection "robustness score" identifying which detections are stable and which are fragile near the decision boundary.

## Key Findings (Simulated Data)

- Model is **miscalibrated** (ECE = 0.149) — high-confidence detections are less reliable than their scores suggest
- Weakest performance under **overcast weather** (F1=0.896) and at **night** (F1=0.904)
- **10.6%** of detections are fragile under input perturbation
- **65.5%** of detections near the decision boundary (0.4–0.6 confidence) are fragile
- Recommends a two-threshold system: accept >0.7, reject <0.3, flag middle range for human review

## Usage

```bash
pip install numpy matplotlib pandas scikit-learn
python detection_analysis.py
```

Outputs saved to `output/`:
- `simulated_detections.csv` — full detection dataset
- `calibration_curve.png` — confidence vs accuracy plot
- `performance_breakdown.png` — metrics by condition
- `sensitivity_analysis.png` — Monte Carlo robustness analysis
- Summary report printed to console

## Adapting to Real Data

Replace `generate_detection_data()` with a function that loads actual mission data into the same DataFrame format:

| Column | Type | Description |
|--------|------|-------------|
| `confidence` | float | Model confidence score (0–1) |
| `ground_truth` | int | 1 = real target, 0 = false positive |
| `target_class` | str | Detection class (vessel, fire, etc.) |
| `time_of_day` | str | day / night |
| `terrain` | str | ocean / land |
| `weather` | str | clear / cloudy / overcast |
| `latitude` | float | Detection latitude |
| `longitude` | float | Detection longitude |

The entire analysis pipeline runs unchanged on any dataset with this schema.

## Author

Luka Silbernagel — Princeton University, ORFE '28
