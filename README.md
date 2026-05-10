# DNN Image Ambiguity Analysis (CIFAR-10H)

## Project Overview
This project investigates how deep neural networks handle inherent visual ambiguity and human disagreement in image classification. By leveraging the **CIFAR-10H** dataset, the project shifts away from traditional "hard" one-hot categorical labels and trains models directly on **soft label distributions** that reflect the consensus and disagreements among multiple human annotators.

## Dataset and Preprocessing
- **Dataset**: CIFAR-10H, containing 10,000 images from the CIFAR-10 test set annotated by multiple humans.
- **Labels**: Probability distributions over the 10 classes, encapsulating the degree of human uncertainty for each image.
- **Exploratory Data Analysis (EDA)**: Computed prediction entropies across classes and graphed a soft confusion matrix to reveal which visual categories humans naturally confuse the most.

## Model Architecture
- **Backbone**: Modified ResNet-18 specially tailored for 32x32 images.
  - The aggressive 7x7 convolution is replaced with a 3x3 convolution (with stride 1) to preserve fine spatial details.
  - The initial Max Pooling layer is replaced with an Identity layer to prevent excessive spatial downsampling.
  - The fully connected prediction head outputs 10 logits that are processed into probability distributions.

## Methods & Techniques

### Loss Engineering
Since the labels are soft distributions rather than singular targets, three distinct loss functions were implemented and evaluated for distribution matching:
1. **KL Divergence Loss**: The standard asymmetric approach for matching distributions. It heavily penalizes false confidence in incorrect prediction categories.
2. **Jensen-Shannon Divergence (JSD) Loss**: A symmetric implementation that bounds the loss and produces more robust, stable gradients by treating both distributions fairly compared to their midpoint.
3. **Custom Composite Entropy Loss**: A novel task-specific loss function containing an entropy regularization term:
   $$ \mathcal{L} = \lambda_1 \cdot \text{KL}(p \parallel q) + \lambda_2 \cdot |\mathcal{H}(p) - \mathcal{H}(q)| $$
   This uniquely forces the network to not only align with the average label distribution but also exactly match the expected level of human uncertainty (entropy).

### Training & Fine-Tuning Pipeline
- Scripts feature standard training pipelines using Adam/CosineAnnealing schedulers.
- Models predict softmax probabilities directly optimized against the soft label targets.

### Evaluation & Analytics
- **Core Metrics**: Extending beyond single-class Accuracy, analyzing KL divergence, Expected Calibration Error, and Top-K retrieval on soft distributions.
- **Failure Analysis**: Identifying specific categories and images where the network's internal ambiguity severely misaligns with human annotators (`failure_analysis.py`).
- **Explainability**: Integrated **Grad-CAM** (`gradcam.py`) specifically adapted for ambiguity, allowing visual comparisons of where models focus their attention depending on the chosen loss function.

### Ablation Studies
A battery of ablation experiments (`ablation.py`) tested critical modeling assumptions:
1. **Backbone Initialization Strategies**: Random weights vs. standard CIFAR-10 pretraining vs. full ImageNet transfer learning.
2. **Loss Function Scaling**: Benchmarking KL, JSD, and the new Custom Entropy composite.
3. **Prediction Head Architecture**: Evaluating Linear heads against Multi-Layer Perceptron (MLP) architectures and deep representation configurations.

## Project Structure
- `dataset.py`, `model.py`, `losses.py`: Core machine learning classes (Data loading, ResNet architecture, Loss implementations).
- `pretrain.py`, `finetune.py`, `run_all.py`: End-to-end model training operations.
- `core_evaluation.py`, `metrics.py`, `evaluate.py`: Granular distribution matching assessments.
- `ablation.py`, `plot_ablation.py`: Structural experiment runners and curve plotting.
- `eda_computation.py`, `eda_plots.py`: Dataset level analysis scripts.
- `gradcam.py`, `failure_analysis.py`, `robustness.py`: Interpretability bounds and secondary analytical validation.
- `LOSS_FUNCTIONS_DOCUMENTATION.md`: Mathematical formulations of the distributions generated during Day 5.