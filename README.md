# ML Start hw02 -- Porto Seguro Safe Driver Prediction

> Сорочан Дмитрий · [Telegram](https://t.me/legenda0008) · [Kaggle](https://www.kaggle.com/dmitrysorochan)

Решение соревнования [Porto Seguro’s Safe Driver Prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction).

## Результат

Лучший submission -- `outputs/optuna_submission.csv`.

| Submission | Private Gini | Public Gini |
|---|---:|---:|
| `gnb_submission.csv` | 0.18599 | 0.18858 |
| `rf_submission.csv` | 0.26584 | 0.25979 |
| `logreg_submission.csv` | 0.26683 | 0.26012 |
| `lgbm_submission.csv` | 0.27963 | 0.27292 |
| `equal_submission.csv` | 0.27966 | 0.27298 |
| `score_weighted_submission.csv` | 0.27976 | 0.27308 |
| **`optuna_submission.csv`** | **0.28134** | **0.27471** |

[Мой профиль -- @dmitrysorochan.](https://www.kaggle.com/dmitrysorochan)

![Kaggle submissions](outputs/scores.png)

В соревновании 5 156 команд, текущий результат находится примерно на границе топ-50% :(((

## Подход

1. EDA: баланс классов, пропуски, распределения, корреляции, train/test shift.
2. Feature engineering: удаление почти константных признаков, обработка категорий и пропусков, проверка PCA.
3. Модели: GaussianNB, Logistic Regression, Random Forest, LightGBM.
4. Optuna с пятифолдовой стратифицированной CV и Normalized Gini.
5. OOF-предсказания и rank blending:
   - равные веса;
   - веса по OOF Gini;
   - веса, подобранные Optuna.

Все обучаемые преобразования находятся внутри CV-пайплайнов.

## Что сработало

- LightGBM стал лучшей одиночной моделью;
- rank blend улучшил результат;
- лучшие веса бленда нашла Optuna;
- удаление только совсем слабых почти константных признаков не ухудшило качество.

## Что не сработало

- GaussianNB дал слабый leaderboard score;
- агрессивное удаление признаков ухудшало CV;
- `PCA(n_components=2)` помог GaussianNB, но ухудшал остальные модели, поэтому в финальных пайплайнах не использовался;
- равные и пропорциональные score веса почти не улучшили LightGBM.

Топовые решения активно использовали нейросети и denoising autoencoders. Здесь решение намеренно собрано только на классическом ML.

## Структура

```text
.
├── 00_EDA.ipynb
├── 01_feature_engineering.ipynb
├── 03_models.ipynb
├── data/
│   ├── download_data.py
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── outputs/
│   ├── scores.png
│   ├── gnb_submission.csv
│   ├── logreg_submission.csv
│   ├── rf_submission.csv
│   ├── lgbm_submission.csv
│   ├── equal_submission.csv
│   ├── score_weighted_submission.csv
│   └── optuna_submission.csv
├── statement/
├── .gitignore
└── README.md
```

CSV с исходными данными и локальные логи не хранятся в Git.

## Запуск

```bash
python3 -m pip install -U \
    numpy pandas matplotlib seaborn scipy \
    scikit-learn lightgbm optuna kaggle jupyter
```

Скачать данные:

```bash
python3 data/download_data.py
```

Запустить ноутбуки по порядку:

```text
00_EDA.ipynb
01_feature_engineering.ipynb
03_models.ipynb
```

Финальные submissions сохраняются в `outputs/`. Optuna trials, OOF-предсказания и графики сохраняются в `.local/` и игнорируются Git.
