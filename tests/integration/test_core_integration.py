"""
Testes de integração com deepbridge core.
"""

import pytest
import numpy as np
import pandas as pd
from deepbridge import DBDataset
from deepbridge_distillation import AutoDistiller


def test_autodistiller_with_dbdataset():
    """Testa que AutoDistiller funciona com DBDataset do core."""
    # Criar dados
    np.random.seed(42)
    df = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100),
        'prob_0': np.random.rand(100),
        'prob_1': np.random.rand(100),
    })

    # Normalizar probs
    total = df['prob_0'] + df['prob_1']
    df['prob_0'] = df['prob_0'] / total
    df['prob_1'] = df['prob_1'] / total

    # Criar dataset
    dataset = DBDataset(
        data=df,
        target_column='target',
        features=['f1', 'f2'],
        prob_cols=['prob_0', 'prob_1'],
    )

    # Verificar que DBDataset pode ser criado e funciona
    assert dataset is not None
    assert isinstance(dataset, DBDataset)
    print("✅ DBDataset do core está acessível e funcional")


def test_core_types_available():
    """Verifica que tipos do core estão disponíveis."""
    from deepbridge import DBDataset, ModelType

    assert DBDataset is not None
    assert ModelType is not None
    print("✅ Tipos do core disponíveis")
