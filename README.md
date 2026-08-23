# ML Start hw03 — тема 24: архитектура MLP, глубина vs ширина

> Сорочан Дмитрий · [Telegram](https://t.me/legenda0008)

## Задача

Тема 24 из ДЗ №3:

> **Архитектура имеет значение: глубина vs ширина.** Исследовать, как число слоёв и нейронов влияет на аппроксимацию. Сравнить shallow vs deep сети при одинаковом числе параметров. Построить heatmap «ширина × глубина → качество». Датасеты: MNIST, Fashion-MNIST, синтетические функции.

Общая часть ДЗ требует пройти полный ML-pipeline, сравнить классические модели и MLP, визуализировать результаты и оформить PDF-отчёт. В качестве классических baseline дополнительно используются Logistic Regression, Random Forest и LightGBM.

## Что сделано

1. Подробный EDA для **MNIST** и **Fashion-MNIST**: баланс классов, примеры, средние изображения классов, распределение пикселей, яркость, pixel variance, расстояния между class centroids, PCA.
2. Простая MLP на PyTorch: только `Linear + ReLU`, без residual connections, BatchNorm и scheduler'ов.
3. Широкий architecture sweep:
   - `depth = [1, 2, 3, 5, 10, 20, 50, 100]`;
   - `width = [8, 16, 32, 64, 128, 256]`.
4. Heatmap `width × depth → accuracy` и `width × depth → macro-F1` отдельно для MNIST и Fashion-MNIST.
5. Графики качества по глубине, ширине и числу параметров.
6. Learning curves для shallow, medium, deep и very deep MLP.
7. Отдельный **parameter-matched** эксперимент при бюджетах примерно 50k / 150k / 500k параметров.
8. Финальная MLP выбирается только по validation, после чего test используется на финальном этапе.
9. Сравнение с Logistic Regression, Random Forest и LightGBM по `accuracy`, `macro-F1`, `balanced accuracy` и времени обучения.
10. Три синтетические функции, отдельная heatmap и визуальное сравнение качества аппроксимации.
11. Подробное обсуждение теоремы об универсальной аппроксимации и того, почему она не означает «одного слоя всегда достаточно на практике».
12. LaTeX-отчёт в `report/report.tex`, оформленный в том же компактном стиле, что приложенный шаблон.

## Методология и защита от leakage

- Official test MNIST/Fashion-MNIST не используется в EDA и architecture selection.
- Official train делится на train/validation со стратификацией и фиксированным `SEED = 143`.
- Нормализация для MLP считается только по train-части.
- Все architecture experiments получают один и тот же split и один и тот же training protocol.
- В sweep фиксированы optimizer, learning rate и число эпох; early stopping не используется.
- В parameter-matched эксперименте width выбирается **только по формуле числа параметров**, а не по validation score.
- Классические модели сравниваются на той же train-подвыборке и том же validation, что и MLP sweep.
- Финальная test-метрика считается после выбора архитектуры по validation.

## Структура

```text
.
├── 00_EDA.ipynb
├── 01_architecture_research.ipynb
├── 02_classical_models.ipynb
├── 03_synthetic_functions.ipynb
├── report/
│   ├── figures/
│   ├── report.tex
│   └── report.pdf
├── .gitignore
├── requirements.txt
└── README.md
```

Весь ML-код находится непосредственно в ноутбуках. Отдельных `src/`, `scripts/`, `tests/` и служебной архитектуры нет.

Датасеты не коммитятся в репозиторий: `torchvision` автоматически скачивает MNIST и Fashion-MNIST в `~/.cache/mlstart_hw03`. Синтетические данные генерируются в ноутбуке.

## Запуск на Linux

### 1. Создать окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Для CUDA PyTorch при необходимости лучше поставить сборку с официальной страницы PyTorch, соответствующую вашей версии CUDA, а затем установить остальные зависимости.

### 2. Запустить Jupyter

```bash
jupyter lab
```

Ноутбуки запускаются по порядку:

```text
00_EDA.ipynb
01_architecture_research.ipynb
02_classical_models.ipynb
03_synthetic_functions.ipynb
```

`01_architecture_research.ipynb` — самый тяжёлый: полный sweep включает 96 MLP-конфигураций только в основной сетке, затем learning curves и parameter-matched эксперименты. GPU рекомендуется, но код автоматически работает и на CPU.

### 3. Собрать PDF-отчёт

После выполнения ноутбуков графики автоматически сохраняются в `report/figures/`.

На Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-lang-cyrillic
cd report
pdflatex report.tex
pdflatex report.tex
```

## Результаты

После полного запуска численные результаты сохраняются рядом с отчётом:

- `report/mnist_architecture_results.csv`;
- `report/fashion_architecture_results.csv`;
- `report/mnist_parameter_matched.csv`;
- `report/fashion_parameter_matched.csv`;
- `report/mlp_final_results.csv`;
- `report/model_comparison.csv`;
- `report/synthetic_grid_results.csv`;
- `report/synthetic_comparison.csv`.

Последняя ячейка `03_synthetic_functions.ipynb` также собирает основные метрики в `report/results.tex`; `report/report.tex` автоматически подхватывает этот файл при следующей компиляции.

Ключевые визуализации:

- `mnist_accuracy_heatmap.png`, `fashion_accuracy_heatmap.png`;
- learning curves train/validation loss и accuracy;
- accuracy vs depth / width / parameter count;
- parameter-matched curves;
- MLP vs Logistic Regression / Random Forest / LightGBM;
- confusion matrices;
- синтетические функции и их MLP-аппроксимации.

Численные выводы нужно читать из выполненных CSV/ноутбуков, а не переносить между запусками вручную. Это важно, потому что итог зависит от устройства, версии библиотек и того, был ли выполнен полный sweep.

## Как интерпретировать эксперимент

Обычная heatmap и parameter-matched эксперимент отвечают на разные вопросы.

Если глубокая сеть лучше при фиксированной ширине, это ещё не доказывает преимущество глубины: она одновременно может иметь намного больше параметров. Более сильный аргумент появляется только если deep-сеть выигрывает и при сопоставимом parameter budget.

С другой стороны, depth = 50–100 для plain MLP без residual connections может оптимизироваться хуже. Это тоже содержательный результат: универсальная аппроксимация говорит о **существовании представления**, но не обещает, что произвольная глубокая архитектура будет легко обучаться.

## Отчёт

Исходник: [`report/report.tex`](report/report.tex).  
PDF: [`report/report.pdf`](report/report.pdf).

Отчёт рассчитан на 3–8 страниц и содержит постановку задачи, EDA, методологию, теорию, описание экспериментов, classical baselines, synthetic experiment и правила интерпретации результатов.
