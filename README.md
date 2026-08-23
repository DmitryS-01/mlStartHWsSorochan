# ML Start hw03 -- глубина vs ширина MLP

> Сорочан Дмитрий · [Telegram](https://t.me/legenda0008)

**Тема 24:** исследование влияния **глубины и ширины MLP** на `MNIST`, `Fashion-MNIST` и синтетических функциях.

## Что сделано

- подробный EDA двух корпусов: баланс классов, примеры, распределения пикселей, variance maps, boxplots, correlation matrix, PCA, class centroids и распределения engineered features по классам;
- 5 engineered image-features: `mean_brightness`, `pixel_std`, `ink_density`, `center_x`, `center_y`;
- единый sweep `depth × width` с `WIDTHS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]`; ширина в том же эксперименте выходит далеко за 784 входных пикселя;
- heatmap `width × depth -> balanced accuracy / macro-F1` и графики качества по глубине/ширине;
- learning curves с train + validation loss на одном графике и train/val balanced accuracy;
- shallow vs deep при одинаковом бюджете параметров;
- 5-fold StratifiedKFold выбранной MLP как проверка устойчивости;
- weight decay как фиксированная регуляризация MLP;
- классика: `GaussianNB`, `Logistic Regression`, `Random Forest`;
- для классики: GridSearchCV на 5 стратифицированных фолдах и внешний hold-out validation;
- balanced accuracy + macro-F1 + macro-precision + macro-recall;
- большие normalized confusion matrix и автоматический анализ top-confusions;
- feature importance для Random Forest с настоящими именами пикселей и engineered features;
- синтетика: почти линейная, полиномиальная, периодическая и Heaviside;
- для каждой синтетической функции строится MSE heatmap `width × depth`;
- для deep `Tanh` используется одинаковая Xavier initialization: сравнение больше не ломается из-за схлопывания сигнала на старте;
- blending: equal-weight и metric-weighted soft voting;
- таблицы корреляций probability predictions и индикаторов ошибок моделей;
- LaTeX/PDF-отчет.

## Методология

Главная метрика -- **balanced accuracy**. На `MNIST` и `Fashion-MNIST` классы почти сбалансированы, но одна основная метрика делает сравнение единообразным. Дополнительно считаются `macro-F1`, `macro-precision` и `macro-recall`.

Для большого MLP sweep используется один фиксированный stratified train/validation split. Это осознанный компромисс: повторять всю большую сетку на пяти фолдах слишком дорого. После выбора лучшая архитектура отдельно проверяется 5-fold CV. Все доступные точки единого width/depth sweep участвуют в выборе кандидата; комбинации больше 20 млн параметров отмечаются как пропущенные до создания модели, чтобы эксперимент не превращался в тест RAM. Official test не участвует ни в sweep, ни в CV; после выбора архитектура переобучается на всей official train-части и test оценивается один раз.

Для классики внешний validation отделяется до настройки моделей. GridSearchCV работает только внутри train на `StratifiedKFold(n_splits=5)`. Scaling Logistic Regression находится внутри `Pipeline`, поэтому scaler не видит validation fold заранее. `GridSearchCV` запускается последовательно (`SEARCH_N_JOBS=1`), а параллелизм остается только внутри Random Forest: это убирает nested parallelism и лишние пики памяти, не меняя пространство гиперпараметров.

Для blending equal-weight вариант не настраивается вообще, а metric-weighted веса пропорциональны CV balanced accuracy, полученной на train. По hold-out validation веса не подбираются. При этом сами blend-метрики считаю **exploratory**: этот validation раньше использовался для выбора MLP-архитектуры, поэтому он не является независимым test ансамбля. Финальная независимая оценка остается official test из `01`.

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

В рабочем проекте каталог `hws/` лежит внутри `/mlStart`. Kernel может быть запущен из `/mlStart`: ноутбуки сами находят `/mlStart/hws` и сохраняют результаты в `/mlStart/hws/output`.

## Порядок запуска

```text
00_EDA.ipynb
01_MLP_architecture.ipynb
02_classical_models.ipynb
03_synthetic_mlp_approximation.ipynb
04_blending.ipynb
```

`04` зависит от probabilities и CV-таблиц, сохраненных `01` и `02`. После изменений в архитектурном sweep, классике или синтетике лучше запускать ноутбуки именно в этом порядке, чтобы в `output/` не оставались результаты старой конфигурации.

## Запуск на Linux

Из `/mlStart`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r hws/requirements.txt
jupyter lab
```

Для сборки отчета:

```bash
cd hws/report
pdflatex report.tex
pdflatex report.tex
```

## Что изменилось в последних экспериментах

### Более широкие MLP

Основной sweep сразу использует ширины от 8 до 2048 и глубины до 100. Чтобы не создавать заведомо многогигабайтные модели, сочетания свыше 20 млн параметров пропускаются до инициализации и остаются пустыми точками на heatmap. Так широкие сети сравниваются в том же протоколе, но Jupyter не падает на `depth=100, width=2048`.

### Feature importance

В классическом ноутбуке 789 признаков: 784 пикселя + 5 engineered features. Importance подписана как `pixel_row_col`, `mean_brightness`, `pixel_std`, `ink_density`, `center_y`, `center_x`. Коррелированные engineered features оставлены: для деревьев это допустимо, но importance таких признаков надо интерпретировать группой, потому что она может распределяться между ними.

### Blending

Отсутствие прироста на `MNIST` -- понятный возможный результат. В сохраненном прогоне probability correlations между `LR`/`RF`/`MLP` высокие (примерно `0.94--0.95`), поэтому модели во многом повторяют друг друга. На MNIST лучшая одиночная `MLP` осталась сильнее простого blend; на `Fashion-MNIST` equal/metric blend дал небольшой прирост. Metric-weighted веса считаются по train-CV, а не подгоняются под hold-out validation.

## Чек-лист задания

### Обязательная часть

- [x] описание датасетов, размеров, типов данных и target;
- [x] визуальный EDA: баланс классов, примеры, распределения, boxplot, correlation matrix, variance maps, centroids, PCA и class-conditional distributions engineered features;
- [x] обработка особенностей данных: пропусков/категорий нет, редкие изображения не удаляются без основания;
- [x] feature engineering: 5 новых image-level признаков;
- [x] минимум 3 классических алгоритма: GaussianNB, Logistic Regression, Random Forest;
- [x] подбор гиперпараметров GridSearchCV;
- [x] не менее 5 фолдов CV;
- [x] обоснование MLP architecture и большой depth/width sweep;
- [x] train/val learning curves;
- [x] регуляризация MLP через weight decay;
- [x] общая таблица качества и времени;
- [x] confusion matrix и содержательный error analysis;
- [x] test не используется для выбора архитектуры или blend-весов.

### Тема 24

- [x] MNIST;
- [x] Fashion-MNIST;
- [x] синтетические функции;
- [x] влияние depth и width;
- [x] heatmap `width × depth -> quality`;
- [x] единый sweep включает ширины 1024 и 2048, то есть больше 784 входных признаков;
- [x] shallow vs deep при сопоставимом числе параметров;
- [x] MSE heatmap для всех выбранных синтетических функций.

### Бонусы

- [x] feature importance + анализ top-features;
- [x] blending нескольких моделей;
- [x] equal-weight и metric-weighted blending;
- [x] корреляции predictions/errors для анализа потенциальной пользы ансамбля;
- [x] error analysis по confusion matrix;
- [x] дополнительный parameter-matched ablation;
- [x] Heaviside как отдельный разрывный пример;
- [x] GitHub-ready README + requirements.

## Воспроизводимость и отсутствие leakage

- `SEED = 143` фиксирован для NumPy, Python random, PyTorch, split и CV;
- train/validation split одинаков для сравниваемых архитектур и классических моделей;
- official test не используется при выборе depth/width, hyperparameters или blend weights;
- GridSearchCV обучается только на train;
- StandardScaler находится внутри Pipeline;
- metric-based blending вычисляет численные веса из train-CV scores; MLP-архитектура при этом уже выбрана по hold-out validation, поэтому blend-оценка явно помечена как exploratory;
- validation probabilities разных моделей дополнительно проверяются на одинаковый порядок `y_true` перед blending;
- новые synthetic initialization/LR одинаковы для всех depth/width конфигураций;
- фиксированные seed обеспечивают воспроизводимый протокол; точное побитовое совпадение классификационных MLP на разных CPU/CUDA/MPS backend не гарантируется библиотеками. Синтетический notebook специально работает на CPU для стабильности большого последовательного sweep.

## Что лежит в `output/`:

- `mnist_val_balanced_accuracy_heatmap.png`, `fashion_val_balanced_accuracy_heatmap.png`;
- `*_learning_curve_loss.png`;
- `*_parameter_matched_balanced_accuracy.png`;
- `mlp_cv_results.csv`;
- `*_classical_results.csv`;
- `*_random_forest_feature_importance.png`;
- `model_comparison.csv`;
- `*_mlp_confusion_matrix.png`, `*_best_classical_confusion_matrix.png`;
- `blending_results.csv`, `blending_weights.csv`;
- `*_prediction_correlations.png`;
- `synthetic_mse_heatmaps_all.png` и отдельные `synthetic_mse_heatmap_<function>.png`;
- `synthetic_approximation_heaviside.png`.
