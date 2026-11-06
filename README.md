# psordinal

[![PyPI](https://img.shields.io/badge/pypi-psordinal-%23ececec.svg?style=flat-square&logo=pypi&logoColor=1f73b7)](https://pypi.org/project/psordinal/)

**psordinal** (Paul77ms's Ordinal Classification) is a Python library
implementing ordinal classification as a cascade of binary classifiers, wrapping
any scikit-learn-compatible estimator.

---

## What ordinal classification is for

Ordinal data has categories with a meaningful **order** but no fixed distance
between them — education level (`High School` < `College` < `Graduate`),
satisfaction (`Poor` < `Fair` < `Good` < `Excellent`), severity grades
(`Grade 1` < `Grade 2` < `Grade 3`).

A plain multi-class classifier throws the order away. Predicting `Grade 1` when
the truth is `Grade 3` is treated as exactly as wrong as predicting `Grade 2` —
which is not how the problem works in practice.

`psordinal` decomposes the ordinal problem into a chain of binary questions
("is it above grade 1?", "is it above grade 2?", …) and recombines the
probabilities, so the class ordering is modelled explicitly.

**Reference reading:**
- [Simple trick to train an ordinal regression with any classifier](https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c)
- [A Simple Approach to Ordinal Classification](https://link.springer.com/chapter/10.1007/3-540-44795-4_13) (Frank & Hall, 2001)

## Install

```bash
pip install psordinal
```

Pinned to a version:

```bash
pip install psordinal==0.1
```

## Quick start

```python
from psordinal import OrdinalClassifier
from xgboost import XGBClassifier

ordinal_clf = OrdinalClassifier(
    clfs=[XGBClassifier] * 5,
    clfs_args=[
        {'objective': 'binary:logistic', 'tree_method': 'hist',
         'random_state': 42, 'enable_categorical': True},
        {'objective': 'binary:logistic', 'tree_method': 'gpu_hist',
         'random_state': 0,  'enable_categorical': True},
        {'objective': 'binary:logistic', 'tree_method': 'gpu_hist',
         'random_state': 7,  'enable_categorical': True},
        {'objective': 'binary:logistic', 'tree_method': 'gpu_hist',
         'random_state': 123, 'enable_categorical': True},
        {'objective': 'binary:logistic', 'tree_method': 'gpu_hist',
         'random_state': 20, 'enable_categorical': True},
    ],
    reverse_classes=False,
)

ordinal_clf.fit(X_train, y_train, categorical_variables)
predictions = ordinal_clf.predict(X_val)
```

A single classifier class works too. Registering **several** applies soft voting
across them.

| Option | Values | Meaning |
| --- | --- | --- |
| `reverse_classes` | `False` *(default)* | Learn in natural order — `Class 0 → 1 → 2 → 3` |
| | `True` | Learn in reverse — `Class 3 → 2 → 1 → 0` |
| `conditional` | `True` *(default)* | Model dependencies between classes, capturing the ordinal relationship |
| | `False` | Treat each class independently |

## API

```python
# fit
ordinal_clf.fit(X_train, y_train, categorical_variables)

# fit with a validation set and early stopping
ordinal_clf.fit_eval(X_train, y_train, X_val, y_val,
                     early_stopping_round, categorical_variables)

# predict
proba = ordinal_clf.predict_proba(X_val, conditional)
preds = ordinal_clf.predict(X_val, conditional)

# interpretability
feature_importance = ordinal_clf.get_feature_importances(feature_names)
shap_values        = ordinal_clf.shap_values(X)
```

## Command line

```bash
psordinal --clfs XGBClassifier \
          --clfs_args '{"objective": "binary:logistic"}' \
          --reverse_classes False \
          --fit \
          --train_data "[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]" \
          --train_labels "[0,1,2,3]" \
          --cat_cols None \
          --predict \
          --test_data "[[4,5,6],[7,8,9]]"
```

| Flag | Description |
| --- | --- |
| `--clfs` | Classifier classes to use (`XGBClassifier`, `CatBoostClassifier`, `LGBMClassifier`) |
| `--clfs_args` | Hyperparameters per classifier, as JSON strings |
| `--reverse_classes` | Learn the class order in reverse |
| `--fit` | Train on `--train_data` / `--train_labels` |
| `--train_data` | NumPy-style array literal or a CSV path |
| `--train_labels` | NumPy-style array literal or a CSV path |
| `--cat_cols` | Categorical columns, e.g. `['A','B','C']` |
| `--predict` | Run prediction after fitting |
| `--test_data` | NumPy-style array literal or a CSV path |

## Results

Accuracy ignores ordering, so it cannot show whether a model respects class
order. **Quadratic Weighted Kappa (QWK)** penalises predictions in proportion to
how far off they are, which is the behaviour that matters here.

Five-fold comparison on the same data:

| Model | Fold mean QWK |
| --- | --- |
| **Ordinal Classifier** | **0.3858** |
| Simple Classifier | 0.3181 |

<details>
<summary>Per-fold scores</summary>

**Ordinal Classifier**

Fold 1 <img width="100" src="https://github.com/user-attachments/assets/309907fb-cd01-426c-b2bf-e0071717318e" />
Fold 2 <img width="100" src="https://github.com/user-attachments/assets/d88b0326-bfa6-439a-8407-3d65a8a5ea9c" />
Fold 3 <img width="100" src="https://github.com/user-attachments/assets/d9aeeeae-8965-4fee-8b32-eb2722b4afa7" />
Fold 4 <img width="100" src="https://github.com/user-attachments/assets/3d4c7244-709b-4113-ac2d-1c858e87115f" />
Fold 5 <img width="100" src="https://github.com/user-attachments/assets/e4c73acd-f410-473b-9198-6726c330bbdd" />

**Simple Classifier**

Fold 1 <img width="100" src="https://github.com/user-attachments/assets/788e6ab7-8cf5-4067-8ed9-5c069834c0fa" />
Fold 2 <img width="100" src="https://github.com/user-attachments/assets/4c97aa53-7a29-4c2a-9807-b73de602c5c9" />
Fold 3 <img width="100" src="https://github.com/user-attachments/assets/3f86a380-60f4-4f13-9c00-2adf7d740ff9" />
Fold 4 <img width="100" src="https://github.com/user-attachments/assets/66edf3fc-ce89-4365-a318-9b89562c7b44" />
Fold 5 <img width="100" src="https://github.com/user-attachments/assets/7268232c-cb49-49c9-b4e1-a3517b2d4d7d" />

</details>

The ordinal decomposition raises QWK from 0.3181 to 0.3858 — it recovers the
class-order information that a flat classifier discards.

## Project layout

```
psordinal/
├── psordinal/
│   ├── __init__.py      exports OrdinalClassifier
│   ├── main.py          CLI entry point
│   └── module.py        OrdinalClassifier implementation
├── requirements.txt
├── setup.py
└── README.md
```

## Development

```bash
git clone https://github.com/paulms77/psordinal.git
cd psordinal

pip install -r requirements.txt
pip install -e .
```

Build and publish:

```bash
python setup.py sdist bdist_wheel          # source + wheel
pip install dist/psordinal-0.1-py3-none-any.whl --force-reinstall
twine upload dist/*
```

## Contributing

Pull requests are welcome — please open an issue first to discuss the proposed
change.

## License

MIT — see [LICENSE](LICENSE).
