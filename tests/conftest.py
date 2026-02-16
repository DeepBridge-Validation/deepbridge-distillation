"""
Fixtures comuns para testes de deepbridge-distillation.
"""

import pytest
import numpy as np
import pandas as pd
from deepbridge import DBDataset  # ← Do deepbridge core


@pytest.fixture
def sample_data():
    """Dataset de exemplo para testes."""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randint(0, 2, n_samples),
        'prob_0': np.random.rand(n_samples),
        'prob_1': np.random.rand(n_samples),
    })

    # Normalizar probabilidades
    total = df['prob_0'] + df['prob_1']
    df['prob_0'] = df['prob_0'] / total
    df['prob_1'] = df['prob_1'] / total

    return df


@pytest.fixture
def sample_dataset(sample_data):
    """DBDataset para testes."""
    return DBDataset(
        data=sample_data,
        target_column='target',
        features=['feature1', 'feature2', 'feature3'],
        prob_cols=['prob_0', 'prob_1'],
    )


@pytest.fixture
def simple_model():
    """Modelo simples para testes."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    return model
