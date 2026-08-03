from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION = "porto-seguro-safe-driver-prediction"
DATA_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = [
        file_name
        for file_name in REQUIRED_FILES
        if (DATA_DIR / file_name).exists()
    ]

    if len(existing_files) == len(REQUIRED_FILES):
        print("Все нужные файлы уже скачаны:")
        for file_name in REQUIRED_FILES:
            print(f"  - {DATA_DIR / file_name}")
        return

    kaggle_command = shutil.which("kaggle")

    if kaggle_command is None:
        raise SystemExit(
            "Команда kaggle не найдена. Установи CLI:\n"
            f"  {sys.executable} -m pip install -U kaggle"
        )

    print(f"Скачиваю данные соревнования {COMPETITION}...")

    subprocess.run(
        [
            kaggle_command,
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(DATA_DIR),
            "--force",
        ],
        check=True,
    )

    archives = list(DATA_DIR.glob("*.zip"))

    for archive_path in archives:
        print(f"Распаковываю {archive_path.name}...")

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(DATA_DIR)

        archive_path.unlink()

    missing_files = [
        file_name
        for file_name in REQUIRED_FILES
        if not (DATA_DIR / file_name).exists()
    ]

    if missing_files:
        missing = ", ".join(missing_files)
        raise SystemExit(
            f"Не удалось найти после скачивания: {missing}.\n"
            "Проверь авторизацию Kaggle и принятие правил соревнования."
        )

    print("\nГотово:")
    for file_name in REQUIRED_FILES:
        file_path = DATA_DIR / file_name
        size_mb = file_path.stat().st_size / 1024**2
        print(f"  - {file_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
