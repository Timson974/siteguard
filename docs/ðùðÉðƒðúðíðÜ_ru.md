# SiteGuard — как запустить (пошагово, Windows)

## Что из чего состоит

| Папка / файл | Что это | Простыми словами |
|---|---|---|
| `notebooks/train_yolov8_colab.ipynb` | Обучение модели | Открываете в Google Colab, нажимаете «выполнить все», получаете `best.pt` и метрики |
| `api/` | REST API на FastAPI | Принимает картинку → прогоняет через модель → сохраняет нарушения в базу |
| `api/app/rules.py` | Правила | «NO-Hardhat» или «NO-Safety Vest» = нарушение |
| `api/app/stats.py`, `sql/analytics.sql` | SQL-аналитика | Нарушения по камерам, по дням, по часам смены |
| `dashboard/` | Дашборд Streamlit | Графики + вкладка «загрузить своё фото» |
| `scripts/simulate_cameras.py` | Симулятор камер | Отправляет фото в API, будто они с 4 камер за последние 14 дней |
| `docker-compose.yml` | Запуск всего | База + API + дашборд одной командой |
| `tests/` + `.github/workflows/ci.yml` | Тесты и CI | GitHub сам проверяет код при каждом изменении |
| `docs/aws_deploy.md` | Деплой в AWS | По желанию, позже |

## Шаг 1. Установить (один раз)

1. **Docker Desktop** — docker.com → Download for Windows. При установке оставьте галочку WSL 2. Перезагрузите компьютер, запустите Docker Desktop и дождитесь зелёного статуса «Engine running».
2. **Git** — git-scm.com.
3. **Python 3.11+** — python.org (галочка «Add to PATH»). Нужен только для симулятора камер.

## Шаг 2. Первый запуск (без обученной модели)

Распакуйте архив, например в `C:\Projects\siteguard`. Откройте PowerShell в этой папке:

```powershell
cd C:\Projects\siteguard
docker compose up --build
```

Первая сборка займёт 5–15 минут: скачивается PyTorch, около 2–3 ГБ. Когда в логах появится `Uvicorn running`, откройте:

- **http://localhost:8000/docs** — API. Можно нажать на `POST /detect` → *Try it out* → загрузить фото.
- **http://localhost:8501** — дашборд.

Пока модель не обучена, API использует стандартную `yolov8n.pt`: она находит людей, но не каски. Дашборд это показывает жёлтым предупреждением.

## Шаг 3. Посмотреть дашборд на демо-данных

В **новом** окне PowerShell:

```powershell
py -m pip install requests
py scripts\simulate_cameras.py --synthetic --n 500
```

В дашборде поставьте галочку **Include synthetic demo data**, и появятся графики. Это **ненастоящие данные**: только чтобы увидеть, как всё работает. Их нельзя приводить как результаты. Удалить:

```powershell
py scripts\simulate_cameras.py --clear-synthetic
```

## Шаг 4. Обучить модель в Google Colab (~1 час)

1. colab.research.google.com → File → Upload notebook → `notebooks/train_yolov8_colab.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**. Если Kaggle попросит ключ, инструкция в ячейке 3.
4. В конце скачается `siteguard_results.zip`. В нём `best.pt`, `metrics.json`, графики и 20 тестовых фото.

## Шаг 5. Подключить обученную модель

1. `best.pt` → в папку `models\`.
2. Фото из `samples\` архива → в папку `samples\` проекта.
3. Перезапустить API:

```powershell
docker compose restart api
```

4. Прогнать фото как будто с камер:

```powershell
py scripts\simulate_cameras.py --images samples --n 200
```

Теперь дашборд показывает **настоящие** результаты модели (галочку synthetic не ставить).

## Шаг 6. Залить на GitHub

1. github.com → New repository → имя **siteguard**, Public, без README (он уже есть).
2. В PowerShell в папке проекта:

```powershell
git init
git add .
git commit -m "SiteGuard: PPE detection API, dashboard, training notebook"
git branch -M main
git remote add origin https://github.com/Timson974/siteguard.git
git push -u origin main
```

`best.pt` и фото в git не попадут (они в `.gitignore`), так и должно быть. Графики из Colab (`results.png`, `confusion_matrix.png`, `class_balance.png`) положите в `docs\` и закоммитьте. На вкладке **Actions** на GitHub должен появиться зелёный CI.

3. Заполните таблицу **Results** в README своими цифрами из `metrics.json`.

## Полезные команды

| Команда | Что делает |
|---|---|
| `docker compose up -d` | Запустить в фоне |
| `docker compose logs -f api` | Смотреть логи API |
| `docker compose down` | Остановить |
| `docker compose down -v` | Остановить и **удалить базу** |
| `docker compose exec db psql -U siteguard -d siteguard` | Войти в базу и выполнять SQL из `sql/analytics.sql` |

## Если что-то не работает

- **«port is already allocated»** — порт 5432, 8000 или 8501 занят. Закройте другую программу или поменяйте левую цифру порта в `docker-compose.yml` (например, `"5433:5432"`).
- **Дашборд пишет «API not reachable»** — API ещё загружается. Подождите минуту и нажмите Refresh.
- **Сборка падает на скачивании** — проверьте интернет и повторите `docker compose build`.
