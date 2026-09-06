# Benchmarking Machine Learning Models on the First Public mmWave-Radar Emotion Dataset

## 1. Team and Responsibilities
- Aayush Shrestha — I handle all parts: dataset, baseline reproduction, models, evaluation, analysis, and write-up.
- Progress and decisions tracked through GitHub commit history and milestone notes in the README.

## 2. Feedback Received and Responses
The proposal was accepted.

## 3. Problem and Motivation
- Emotion recognition is used in healthcare, driver monitoring, and safety, but relies on cameras and wearables that are privacy-hostile and fail in poor lighting.
- mmWave radar is contactless: it reads emotion from breathing and heartbeat, with no face or voice recorded.
- Cai et al. (2026) released the first public mmWave emotion dataset but validated it with only a single SVM (~60-70%), leaving stronger models and honest evaluation open.

## 4. Research Questions and Hypotheses
- RQ1: Do stronger classical models (Decision Tree, Random Forest, XGBoost, small MLP) beat the SVM baseline on the same features?
- RQ2: How much does accuracy drop under cross-subject (leave-one-subject-out) evaluation vs. the paper's within-subject split?
- RQ3: Is the ceiling set by the model or by the noisy labels?
- H1: Stronger models beat SVM only marginally; the limit is weak signal and noisy labels, so cross-subject accuracy drops and the label-noise curve stays flat.

## 5. Related Work
**Cai et al. (2026), "An Emotion Recognition Dataset Using Millimeter Wave Radar and Physiological Reference Signals," Scientific Data.**
- Provides the first public mmWave emotion dataset (radar + PPG + GSR, 15 subjects, SAM labels) with a single SVM baseline.
- I reuse their released features, labels, and SVM protocol as my reproduction target.
- I differ by benchmarking several models and adding cross-subject and label-noise evaluation, which they left as future work.

## 6. Proposed System or Approach
- Build a reproducible benchmark harness: one pipeline, identical splits, seeds, preprocessing, and metrics for every model.
- Reuse from Cai et al.: features, SAM labels, and the SVM baseline.
- Add: Decision Tree, Random Forest, XGBoost, small MLP, plus a leakage-safe evaluation layer.
- Replication target: reproduce the SVM numbers from Table 7 and the per-scale counts from Table 6 (2472, 2760, 2928).
- Out of scope: deep-learning models, new data collection.

## 7. Evaluation Plan
- Mapping: RQ1 = model comparison on identical splits; RQ2 = within- vs. cross-subject gap; RQ3 = accuracy vs. injected label noise (0-40%).
- Data: public feature set — 285 clips x 12 windows; 32 mmWave / 28 PPG / 24 GSR features; three binary targets (valence, arousal, dominance).
- Baseline: reproduced SVM. Varied: model, split protocol, modality (mmWave vs. + PPG/GSR), noise level.
- Metrics: balanced accuracy (%), F1, per-subject spread (mean +/- std); 50% chance line shown. Balanced accuracy leads because classes are imbalanced.
- Rigor: all 15 subjects, >=5 seeds, reported as mean +/- std.
- Plots: model-comparison bar chart, within-vs-cross-subject gap plot, label-noise curve, per-scale confusion matrices.

## 8. Expected Deliverables
- Source code for the benchmark harness (data loading, models, evaluation, plotting).
- Raw per-run results (CSV) and aggregated mean +/- std summary tables.
- Final figures: model comparison, cross-subject gap, label-noise curve, confusion matrices.
- README that reproduces the SVM baseline with one command.
- Short annotated demo video and the final report.

## 9. Timeline and Milestones
- Data loading pipeline, reproduce SVM baseline on the Cai et al. features.
- Add the other models (Decision Tree, RF, XGBoost, MLP), verify they run on identical splits.
- Cross-subject evaluation, measure the within- vs. leave-one-subject-out gap.
- Run the full sweep across seeds, run the label-noise study.
- Analysis, build the figures, error analysis on missed cases.
- Write the report, finalize the demo and code.

## 10. Risks and Mitigations
- The reproduced SVM baseline may not match the paper. I'll report the difference and continue from my own recreated baseline.
- The accuracy ceiling may be very low, with even XGBoost near chance. I'll run the label-noise study and present the result as "the data sets the ceiling, not the model."
- Cross-subject accuracy may vary a lot across subjects. I'll report mean and standard deviation with the per-subject spread, and mark those results as exploratory.

## 11. Reproducibility Plan
- Code: Python 3.11, scikit-learn, XGBoost. Versions pinned in pyproject.toml and uv.lock.
- Data: Cai et al. mmWave emotion dataset (Zenodo DOI 10.5281/zenodo.15825931). Download instructions in the README.
- Reproducibility: fixed random seeds, version-controlled configs. All results generated via a single script.

## 12. References
1. Cai, J., Zhang, X., Pan, Y., Zhou, H. (2026). An Emotion Recognition Dataset Using Millimeter Wave Radar and Physiological Reference Signals. Scientific Data, 13, 820. https://doi.org/10.1038/s41597-026-07159-6
2. Zeng, K., Liu, G. (2023). Emotion Recognition Based on Millimeter Wave Radar. Proceedings of the 2023 3rd International Conference on Bioinformatics and Intelligent Computing (BIC '23), 232-236. https://doi.org/10.1145/3592686.3592728
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
