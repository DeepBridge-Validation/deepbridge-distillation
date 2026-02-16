"""
Example: Basic Model Distillation

Shows how to use AutoDistiller to compress a model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from deepbridge import DBDataset
from deepbridge_distillation import AutoDistiller


# Generate data
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'f1': np.random.randn(n_samples),
    'f2': np.random.randn(n_samples),
    'f3': np.random.randn(n_samples),
    'target': np.random.randint(0, 2, n_samples),
})

# Train teacher model
X = df[['f1', 'f2', 'f3']]
y = df['target']

teacher = RandomForestClassifier(n_estimators=100, random_state=42)
teacher.fit(X, y)

# Get predictions
df['prob_0'] = teacher.predict_proba(X)[:, 0]
df['prob_1'] = teacher.predict_proba(X)[:, 1]

# Create dataset
dataset = DBDataset(
    data=df,
    target_column='target',
    features=['f1', 'f2', 'f3'],
    prob_cols=['prob_0', 'prob_1']
)

# Run distillation
print("Running AutoDistiller...")
distiller = AutoDistiller(
    dataset=dataset,
    output_dir='./distillation_results',
    n_trials=10,
    verbose=True
)

results = distiller.run(use_probabilities=True)

print(f"\n✅ Distillation completed!")
print(f"Best student accuracy: {results['best_accuracy']:.3f}")
print(f"Model saved to: {results['model_path']}")
