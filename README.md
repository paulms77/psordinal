# Welcome to psordinal 👋 (Paul77ms)

### **psordinal** is a Python library that integrates ordinal classification methodology papers.

An Ordinal Classifier is a type of machine learning model designed to hadle ordinal data. Ordinal data refers to categories with a meaningful order but without a fixed distance between them. Examples include education levels (e.g, 'High School', 'College', 'Graduate'), satisfaction ratings (e.g., 'Poor', 'Fair', 'Good', 'Excellent'), or grades (e.g., Grade 1, Grade 2, Grade 3).

순서 분류기는 순서 데이터를 처리하도록 설계된 일종의 기계 학습 모델입니다. 순서형 데이터는 의미 있는 순서가 있지만 범주 사이에 고정된 거리가 없는 범주를 나타냅니다. 예를 들면 교육 수준(예: '고등학교', '대학', '졸업'), 만족도 등급(예: '나쁨', '보통', '좋음', '우수') 또는 성적(예: 1등급)이 있습니다. , 2학년, 3학년).

순서형 분류기 방법론에 대한 자세한 내용은 논문을 참고하세요.

 - https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
 - https://link.springer.com/chapter/10.1007/3-540-44795-4_13

## PyPI ![PyPi](https://img.shields.io/badge/pypi-%23ececec.svg?style=for-the-badge&logo=pypi&logoColor=1f73b7)
[https://pypi.org/psordinal/](https://pypi.org/project/psordinal/)

## Structure

```bash
msordinal/
├── msordinal/
│   ├── __init__.py
│   ├── main.py
│   └── module.py
├── README.md
├── requirements.txt
└── setup.py
```

## Source distribution (sdist) and Binary distribution (bdist_wheel) of Python packages

```bash
python setup.py sdist bdist_wheel
```

## Install the Python wheel located in your local path

```bash
pip install dist/msordinal-0.2-py3-none-any.whl --force-reinstall
```

## How to use Commands

Example)

```bash
msordinal --clfs XGBClassifier --clfs_args '{"objective": "binary:logistic"}' --reverse_classes False --fit --train_data "[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]" --train_labels "[0, 1, 2, 3]" --cat_cols None --predict --test_data "[[4, 5, 6], [7, 8, 9]]"
```

Option

--clfs [Model classes]
설명 : 사용할 분류 모델 클래스를 지정.
입력값 : 모델 클래스 이름 (예: XGBClassifier, CatBoostClassifier, LGBMClassifier)

--clfs_args [Arguments for model classes]
설명 : 각 모델 클래스에 전달할 하이퍼파라미터
입력값 : JSON 문자열 포멧으로 전달.

--fit
설명 : 모델을 학습시키는 옵션 --fit 플래그를 추가하면 지정된 모델이 --train_data와 --train_labels를 학습.

--train_data []
설명 : 학습 데이터를 지정. 학습 데이터는 Numpy 배열 혹은 CSV 파일 경로로 입력.
입력값 : Numpy 배열 혹은 학습 데이터 파일 경로

--train_labels []
설명 : 학습 데이터에 해당하는 레이블을 지정. 레이블 데이터는 Numpy 배열 혹은 CSV 파일 경로로 입력.
입력값 : Numpy 배열 혹은 레이블 데이터 파일 경로

--cat_cols []
설명 : 범주형(카테고리형) 컬럼을 지정.
입력값 : Example) ['A', 'B', 'C'] 리스트로 전달.

--predict
설명 : 학습된 모델을 사용하여 예측을 수행.

--test_data
설명 : 예측에 사용할 테스트 데이터를 지정. 테스트 데이터는 NumPy 배열 또는 CSV 파일 경로로 입력.
입력값 : Numpy 배열 혹은 테스트 데이터 파일 경로

## Publish to PyPI

```bash
twine upload dist/*
```

## Python package dependency management

You can manage versions

```bash
pip freeze > requirements.txt
```

You can install it at once.

```bash
pip install -r requirements.txt
```

## Install Python package

The easiest installation method is as follows

```bash
pip install msordinal
```

## Install Python package for version

```bash
pip install msordinal==0.2
```

## Import Module

```python
from msordinal import OrdinalClassifier
```

## How to use Method

How to initialize OrdinalClassifier

```python
ordinal_clf = OrdinalClassifier(
    clfs = [
        XGBClassifier,
        XGBClassifier,
        XGBClassifier,
        XGBClassifier,
        XGBClassifier
    ],
    
    clfs_args = [
        {'objective': 'binary:logistic', 'tree_method': 'hist', 'random_state': 42, 'use_label_encoder': False, 'enable_categorical': True},
        {'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'random_state': 0, 'use_label_encoder': False, 'enable_categorical': True},
        {'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'random_state': 7, 'use_label_encoder': False, 'enable_categorical': True},
        {'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'random_state': 123, 'use_label_encoder': False, 'enable_categorical': True},
        {'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'random_state': 20, 'use_label_encoder': False, 'enable_categorical': True},
    ],
    reverse_classes = False
)
```

reverse_classes = False (default) learns class relationships in natural order
Example) [Class 0 -> Class 1 -> Class 2 -> Class 3] 

reverse_classes = True learns class relationships in reverse order
Example) [Class 3 -> Class 2 -> Class 1 -> Class 0]

Registration is possible with a single model class, but SoftVoting is applied when registering multiple model classes.

How to use fit

```python
ordinal_clf.fit(X_train, y_train, categorical_variables)
```

How to use fit_eval

```python
ordinal_clf.fit_eval(X_train, y_train, X_val, y_val, early_stopping_round, categorical_variables)
```

How to use predict_proba

```python
predict_proba = ordinal_clf.predict_proba(X_val, conditional)
```

How to use predict

```python
predict = ordinal_clf.predict(X_val, conditional)
```

conditional = True (default) : Capture ordinal relationships between classes by explicitly modeling dependencies

conditional = False : The model processes each class independently without considering dependencies

How to use Feature Importance

```python
feature_importance = model.get_feature_importances(feature_names)
```

How to use Shap Values

```python
shape = model.shap_values(X)
```

## Result
순서를 무시한 단순 정확도(Accuracy)는 순서 정보와 예측 간의 차이를 평가하지 못하나, Quadratic Weighted Kappa(QWK)는 클래스 간의 순서를 고려하여 평가할 수 있습니다. Quadratic Weighted Kappa를 사용하여 평가하였습니다.

Ordinal Classifier

Fold1 <img width="100" alt="image" src="https://github.com/user-attachments/assets/309907fb-cd01-426c-b2bf-e0071717318e" />
Fold2 <img width="100" alt="image" src="https://github.com/user-attachments/assets/d88b0326-bfa6-439a-8407-3d65a8a5ea9c" />
Fold3 <img width="100" alt="image" src="https://github.com/user-attachments/assets/d9aeeeae-8965-4fee-8b32-eb2722b4afa7" />
Fold4 <img width="100" alt="image" src="https://github.com/user-attachments/assets/3d4c7244-709b-4113-ac2d-1c858e87115f" />
Fold5 <img width="100" alt="image" src="https://github.com/user-attachments/assets/e4c73acd-f410-473b-9198-6726c330bbdd" />

**Fold Mean QWK : 0.3858**

Simple Classifier

Fold1 <img width="100" alt="image" src="https://github.com/user-attachments/assets/788e6ab7-8cf5-4067-8ed9-5c069834c0fa" />
Fold2 <img width="100" alt="image" src="https://github.com/user-attachments/assets/4c97aa53-7a29-4c2a-9807-b73de602c5c9" />
Fold3 <img width="100" alt="image" src="https://github.com/user-attachments/assets/3f86a380-60f4-4f13-9c00-2adf7d740ff9" />
Fold4 <img width="100" alt="image" src="https://github.com/user-attachments/assets/66edf3fc-ce89-4365-a318-9b89562c7b44" />
Fold5 <img width="100" alt="image" src="https://github.com/user-attachments/assets/7268232c-cb49-49c9-b4e1-a3517b2d4d7d" />

**Fold Mean QWK : 0.3181**

**Ordinal Classifier는 클래스 간의 순서 정보를 효과적으로 학습하고 이를 예측하는 데 강점을 보였습니다. 이를 확인하기 위해 Quadratic Weighted Kappa (QWK) 점수를 비교한 결과, Ordinal Classifier의 QWK 점수는 0.3858로, Simple Classifier의 QWK 점수인 0.3181을 능가하였습니다. 이는 Ordinal Classifier가 클래스 간의 순서 관계를 더 잘 반영하여 예측의 품질을 향상시켰음을 보여줍니다.**

## Collaborating

Contributions to the msordinal project are welcome via pull requests. Please open an issue to discuss your proposed contribution.

You can clone the repository and then install the library from the local repository

```bash
git clone 
```

```bash
pip install ./msordinal
```
