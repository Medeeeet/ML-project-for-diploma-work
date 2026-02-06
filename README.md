# Driving Style Classification & Accident Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

> **Bachelor Diploma Project:** Developing Machine Learning Algorithms to Assess Driving Style and Predict Accident Risks

Комплексная система машинного обучения для анализа стиля вождения, классификации поведения водителей и прогнозирования рисков дорожно-транспортных происшествий.

---

## Содержание

* [ О проекте](#-о-проекте)
* [ Основные возможности](#-основные-возможности)
* [️ Архитектура проекта](#️-архитектура-проекта)
* [ Быстрый старт](#-быстрый-старт)
* [ Модели машинного обучения](#-модели-машинного-обучения)
* [ Система оценки рисков](#-система-оценки-рисков)
* [ Результаты](#-результаты)
* [️ Использование](#️-использование)
* [ Документация модулей](#-документация-модулей)
* [ Вклад в проект](#-вклад-в-проект)
* [ Лицензия](#-лицензия)

---

## О проекте

Данный проект представляет собой полноценный pipeline машинного обучения для:

* **Анализа поведения водителей** на основе телематических данных
* **Классификации стиля вождения** на 4 категории (Safe, Normal, Aggressive, Risky)
* ️ **Прогнозирования рисков** дорожно-транспортных происшествий
* **Генерации рекомендаций** по улучшению безопасности дорожного движения

### Академический контекст

Проект разработан в рамках дипломной работы бакалавра и демонстрирует:

* Современные подходы к машинному обучению
* Профессиональную архитектуру кода
* Комплексные методы оценки моделей
* Воспроизводимость результатов

---

## Основные возможности

### Многомодельный подход

* Классические ML модели: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
* Глубокое обучение: MLP и LSTM для временных рядов
* Ensemble-подходы для повышения точности

### Предобработка данных

* Обработка пропусков (mean, median, KNN)
* Балансировка классов (SMOTE, class weights)
* Нормализация и масштабирование признаков
* Feature engineering для телематических временных рядов
* Обнаружение и обработка выбросов

### Оценка качества

* Accuracy, Precision, Recall, F1-score
* ROC-AUC и confusion matrix
* Сравнительный анализ моделей

### Оценка риска

* Вероятностный риск-скоринг (0–1)
* Классификация уровней риска
* Генерация рекомендаций по безопасности

---

## ️ Архитектура проекта

```
driving_style_ml/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── risk_scoring.py
│   └── utils.py
├── models/
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── experiments/
├── main.py
├── requirements.txt
└── README.md
```

---

## Быстрый старт

### Требования

* Python 3.8+
* pip
* (Опционально) GPU для DL моделей

### Установка

```bash
git clone https://github.com/yourusername/driving-style-ml.git
cd driving-style-ml
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Запуск

```bash
python main.py
```

Дополнительные опции:

```bash
python main.py --synthetic
python main.py --no-dl
python main.py --quick
python main.py --seed 42
```

---

## Модели машинного обучения

### Классические модели

| Модель              | Назначение                     |
| ------------------- | ------------------------------ |
| Logistic Regression | Базовый линейный классификатор |
| Random Forest       | Ансамбль деревьев              |
| Gradient Boosting   | Последовательный бустинг       |
| XGBoost             | Оптимизированный бустинг       |

### Deep Learning

| Модель | Применение         |
| ------ | ------------------ |
| MLP    | Табличные признаки |
| LSTM   | Временные ряды     |

Все параметры задаются в `src/config.py`.

---

## Система оценки рисков

| Score   | Уровень риска | Интерпретация       |
| ------- | ------------- | ------------------- |
| 0.0–0.3 | LOW           | Безопасное вождение |
| 0.3–0.6 | MEDIUM        | Умеренный риск      |
| 0.6–0.8 | HIGH          | Высокий риск        |
| 0.8–1.0 | CRITICAL      | Критический риск    |

Факторы риска включают скорость, резкие манёвры, торможение, ускорение и другие телематические показатели.

---

## Результаты

Типичные метрики:

| Модель        | Accuracy  | F1        | ROC-AUC   |
| ------------- | --------- | --------- | --------- |
| Random Forest | 0.85–0.92 | 0.84–0.91 | 0.90–0.96 |
| XGBoost       | 0.86–0.93 | 0.85–0.92 | 0.91–0.97 |
| MLP           | 0.83–0.90 | 0.82–0.89 | 0.88–0.95 |
| LSTM          | 0.84–0.91 | 0.83–0.90 | 0.89–0.96 |

Результаты сохраняются в папке `outputs/`.

---

## ️ Использование

```python
from src.data_loader import create_sample_dataset
from src.preprocessing import preprocess_pipeline
from src.train import ModelTrainer
from src.evaluate import evaluate_all_models

# Data
df = create_sample_dataset(1000)
X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_pipeline(df)

# Training
trainer = ModelTrainer()
trainer.train_all_models(X_train, y_train, X_val, y_val)

# Evaluation
_, results = evaluate_all_models(trainer.models, X_test, y_test)
```

---

## Документация модулей

* `config.py` — конфигурация проекта
* `data_loader.py` — загрузка и генерация данных
* `preprocessing.py` — обработка и feature engineering
* `models.py` — ML/DL модели
* `train.py` — обучение
* `evaluate.py` — метрики и сравнение
* `risk_scoring.py` — расчет риск-скора
* `utils.py` — вспомогательные функции

---

## Вклад в проект

Pull requests приветствуются. Идеи для развития:

* Real-time inference
* Web dashboard
* Mobile integration
* Transformer-based модели
* Federated learning

---

## Лицензия

Проект разработан в академических целях как часть дипломной работы бакалавра.

---

## Автор

**Латип Медет**
Бакалаврская дипломная работа
**Astana IT University**
**Год:** 2026

---

<div align="center">

** Если проект был полезен, поставьте звезду! **

Made with ️ for safer roads

</div>
