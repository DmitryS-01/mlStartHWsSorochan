# ML Start hw03 -- глубина vs ширина MLP

> Сорочан Дмитрий · [Telegram](https://t.me/legenda0008)

Тема 24: исследование влияния **глубины и ширины MLP** на MNIST, Fashion-MNIST и синтетических функциях.

## Что сделано

- подробный `EDA` двух корпусов: `histograms`, `boxplots`, `correlation matrix`, `PCA`, `class centroids`;
- 5 engineered image-features для обязательной части задания _(часть из них оказалась коррелированной)_;
- большой sweep `MLP`: `depth = 1, 2, 3, 5, 10, 20, 50, 100`, `width = 8, 16, 32, 64, 128, 256` $\textcolor{red}{\#TODO}$ увеличить ширину;
- heatmap `width × depth -> balanced accuracy / macro-F1`;
- `learning curves` с train + validation loss на одном графике и train/val balanced accuracy;
- shallow vs deep при одинаковом бюджете параметров;
- 5-fold StratifiedKFold для выбранной `MLP` как проверка устойчивости;
- weight decay как фиксированная регуляризация MLP;
- классика: `GaussianNB`, `Logistic Regression`, `Random Forest`, ~~`LightGBM`~~;
- для классики: GridSearchCV на 5 стратифицированных фолдах и внешний validation;
- balanced accuracy + macro-F1 + macro-precision + macro-recall;
- большие normalized confusion matrix и автоматический анализ top-confusions;
- синтетика: почти линейная, полиномиальная, периодическая и Heaviside;
- bonus feature importance `Random Forest`;
- bonus blending: классика и классика + `MLP`;
- LaTeX/PDF-отчет.

## Методология

Главная метрика -- **balanced accuracy**. На `MNIST` / `Fashion-MNIST` классы почти сбалансированы, но единая метрика делает все сравнения согласованными. Дополнительно считаются `macro-F1`, `macro-precision` и `macro-recall`.

Для большого `MLP` sweep используется один фиксированный stratified train/validation split. Test не участвует в выборе архитектуры. После выбора лучшая конфигурация дополнительно проверяется 5-fold CV, затем переобучается на всей official train-части и только после этого один раз оценивается на official test.

Для классики внешний validation отделяется **до** настройки моделей. GridSearchCV работает только внутри train на `StratifiedKFold(n_splits=5)`. Обучаемый scaling Logistic Regression находится внутри `Pipeline`, поэтому fit scaler выполняется только на соответствующем train-fold.

## Структура

```text
hws/
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_MLP_architecture.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_synthetic_mlp_approximation.ipynb
│   └── 04_blending.ipynb
├── output/                 # CSV, PNG, NPZ из ноутбуков
├── report/
│   ├── report.tex
│   └── report.pdf
├── statement/homework3.pdf
├── requirements.txt
└── README.md
```

## Порядок запуска

```text
00_EDA.ipynb
01_MLP_architecture.ipynb
02_classical_models.ipynb
03_synthetic_mlp_approximation.ipynb
04_blending.ipynb
```

`04` зависит от probabilities, сохраненных `01` и `02`. Остальные ноутбуки независимы по коду, но логически лучше запускать именно в указанном порядке.

## Запуск на Linux

Из root-директории:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r hws/requirements.txt
jupyter lab
```

Все результаты сохраняются в `hws/output/`.

Для сборки отчета:

```bash
cd hws/report
pdflatex report.tex
pdflatex report.tex
```

## Чек-лист задания

### Обязательная часть

- [x] описание датасета, размеры, типы данных, target;
- [x] визуальный EDA: баланс классов, примеры, распределения пикселей, variance maps, class centroids, PCA;
- [x] обработка особенностей данных: пропусков/категорий нет, редкие изображения не удаляются как «выбросы» без основания;
- [x] feature engineering: brightness, pixel std, ink density, center X/Y;
- [x] минимум 3 классических алгоритма: фактически 4;
- [x] подбор гиперпараметров GridSearchCV;
- [x] не менее 5 фолдов CV;
- [x] обоснование MLP architecture и большой depth/width sweep;
- [x] train/val learning curves;
- [x] регуляризация MLP через weight decay;
- [x] общая таблица качества и времени;
- [x] confusion matrix и содержательный error analysis;
- [x] test не используется для выбора архитектуры.

### Тема 24

- [x] MNIST;
- [x] Fashion-MNIST;
- [x] синтетические функции;
- [x] влияние depth и width;
- [x] heatmap `width × depth -> quality`;
- [x] shallow vs deep при сопоставимом числе параметров.

### Бонусы

- [x] blending нескольких моделей;
- [x] GitHub-ready README + requirements;
- [x] дополнительное исследование: error analysis + parameter-matched ablation + Heaviside как разрывный пример.
- [x] feature importance LightGBM + анализ top-features.

## Воспроизводимость

Во всех ноутбуках используется `SEED = 143`. Split и `StratifiedKFold` фиксированы. PyTorch, NumPy и Python random получают тот же seed. В GridSearchCV `n_jobs=-1`, LightGBM и Random Forest тоже работают в один процесс -- медленнее, зато без скачков памяти и без различий из-за параллельного выполнения.

## Что смотреть в результатах

После полного запуска основные таблицы лежат в `output/*.csv`, картинки -- в `output/*.png`. В первую очередь:

- `mnist_*_heatmap.png`, `fashion_*_heatmap.png`;
- `*_learning_curve_loss.png`;
- `*_parameter_matched_balanced_accuracy.png`;
- `mlp_cv_results.csv`;
- `*_classical_results.csv`;
- `*_mlp_confusion_matrix.png`, `*_best_classical_confusion_matrix.png`;
- `blending_results.csv`;
- `synthetic_mse_heatmap.png`, `synthetic_approximation_heaviside.png`.
