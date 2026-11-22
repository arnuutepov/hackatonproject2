# Открытые вопросы к правилам и наши решения

В текстах кейса, определениях проверок жюри (20.11.2026) и ответах организаторов (21.11.2026)
некоторые места можно понять по-разному. Для каждого такого места здесь записано:

* что мы приняли и на чём это основано;
* где это в коде;
* какой тест это закрепляет.

Если жюри понимает правило иначе, по этой таблице видно, что поменять и какой тест это покажет.

Источники:

* **кейс** — текст ТЗ;
* **опр.** — определения проверок жюри от 20.11;
* **отв.** — ответы организаторов от 21.11 (их точный текст мы видели только в изложении другой команды, см. DESIGN §14);
* **наше** — решение там, где источники молчат.

## Контракт

| # | Вопрос | Что мы приняли (источник) | Код | Тест |
|---|---|---|---|---|
| 1 | Обязателен ли `episode_id` в ответе и сравнивается ли он? | Необязателен. Значение не сравнивается: ответ без него, с `null` или с чужим ID оценивается так же (отв.) | `grader/reward.py`, `validate_submission` | `test_jury_definitions.py::test_score_episode_ignores_episode_id_value`, `::test_submit_through_adapter_accepts_any_episode_id` |
| 2 | От какого пользователя запускается контейнер? | От `USER` образа, без root (отв.). Образ объявляет `USER 10001:10001`. Если папка репозитория недоступна для записи, результаты пишутся во временную папку | `Dockerfile`, `run.py` `_output_root` | `test_jury_simulation.py::test_installed_package_works_from_a_foreign_directory`, `make docker-check-hardened`, лог `artifacts/logs/check_readonly_nonroot_py311.log` |
| 3 | Как отсчитывается лимит времени 60 с? | С начала `reset`, время политики включено (опр.). Действие после срока завершает эпизод как `truncated` с пустым ответом. Время оценщика после своевременного `submit` агенту не засчитывается (наше) | `env/core.py` | `test_hard_audit.py::test_time_limit_on_the_real_clock`, `::test_time_limit_counts_policy_time_between_steps_on_the_real_clock` |
| 4 | Что делать, если шаги кончились без `submit`? | 100 шагов, `submit` входит в их число. После последнего шага эпизод усекается и оценивается пустой ответ (кейс + наше) | `env/core.py` `MAX_STEPS` | `test_env_contract.py::test_step_limit_truncates_with_empty_answer`, `::test_submit_as_last_allowed_step_counts` |
| 5 | Какие ключи у `info`? | `episode_id`, `mode`, `env_version`, `api_version`, `steps_used` (наше). В закрытом режиме нет метрик и эталона | `judge.py` `_info` | `test_jury_definitions.py::test_contract_check` |
| 6 | Куда `generate_episode` кладёт файлы? | `output_dir/public/` и `output_dir/private/`. Один seed и одна конфигурация дают побайтно тот же результат | `judge.py` `generate_episode` | `test_jury_definitions.py::test_generate_episode_writes_into_output_dir_and_is_deterministic`, `test_golden_hashes.py` |

## Фикстура жюри и оценка

| # | Вопрос | Что мы приняли (источник) | Код | Тест |
|---|---|---|---|---|
| 7 | В каком формате придёт общая фикстура жюри для теста 12? | Формат неизвестен (отв.: одна фикстура на все команды). Загрузчик принимает: `public/`+`private/`, `open/`+`closed/`, плоскую папку и одну вложенную; таблицы `.csv`, `.jsonl` или `.json`; эталон списком или объектом `{issues}`. Неизвестный формат даёт `GradingInputError`, а не неверный счёт (наше) | `grader/episode.py` | `test_jury_fixture_layouts.py` (10 тестов), `test_jury_definitions.py::test_matcher_check_on_a_flat_jury_made_fixture` |
| 8 | Откуда брать допуск τ? | `policy.json`: `amount_tolerance_minor`, число или `{тип: число}` (опр.). Если у ошибки в эталоне свой `tolerance_minor`, он важнее. Без `policy.json` τ = 0 (наше: самое строгое) | `grader/matcher.py` `issue_tolerance`, `grader/episode.py` `tolerance_from_policy` | `test_matcher_metamorphic.py::test_tolerance_edge_is_exact`, `test_jury_fixture_layouts.py` |
| 9 | Какие ID считаются существующими доказательствами? | `row_id` журнала, `invoice_id`, `payment_id` и `doc_id` документов, `fx_id` и ключ курса `дата:из:в` (опр.: каждое доказательство должно существовать). `entity_id` доказательством не считается: это контрагент, а не запись (наше) | `grader/episode.py` `ID_KEYS`, `evidence_universe` | `test_jury_definitions.py::test_evidence_must_exist_when_the_universe_is_known`, `test_matcher_metamorphic.py` (свойство 3) |
| 10 | Как сравнивать ID? | Как точные строки: без обрезки пробелов и без смены регистра. `affected_ids` и `evidence_ids` сравниваются как множества. Лимиты длины списков проверяются до свёртки в множество (опр.) | `grader/matcher.py`, `grader/reward.py` | `test_jury_definitions.py::test_ids_are_exact_strings_no_trimming_or_case_folding` |
| 11 | Какая длина ID допустима? | До 256 кодовых точек; длиннее — структурная ошибка, ответ получает −1 (опр. + наше: запас под любое разумное переименование) | `grader/reward.py` `MAX_ID_LEN` | `test_jury_definitions.py::test_long_ids_up_to_256_code_points_are_valid` |
| 12 | Какие пары выбирать, если максимальных паросочетаний несколько? | Число TP — размер максимального паросочетания один к одному. Пары выбираются лексикографически: ошибки по `issue_id`, каждой — наименьший индекс находки, который ещё позволяет максимум (опр.). На число TP этот выбор не влияет, только на список пар | `grader/matcher.py` `lexicographic_maximum` | `test_matcher_exhaustive.py` (все 74 963 графа до 4×4 против перебора), `test_jury_definitions.py::test_lexicographic_tie_break_among_maximum_matchings` |
| 13 | Чем считаются повторы, неизвестный тип и чужая валюта? | Ложными срабатываниями, а не структурной ошибкой (опр.) | `grader/matcher.py`, `grader/reward.py` | `test_matcher_metamorphic.py` (свойство 2), `test_matcher_reward.py::test_duplicates_give_no_benefit`, `::test_wrong_currency_and_amount_outside_tolerance` |
| 14 | Нужно ли пересортировывать списки эталона после переименования ID? | Нет: порядок `affected_ids` и наборов доказательств в эталоне не важен (опр.: парные проверки) | `generator/validate.py` | `test_jury_definitions.py::test_truth_lists_need_not_be_sorted_after_renaming`, `test_hard_audit.py::test_validator_accepts_unsorted_truth_lists` |
| 15 | Какие поля жюри переименует в парной проверке? | Все ID и все ссылки на них. Ключ курса `дата:из:в` и `episode_id` остаются. Полный список полей — `auditenv/schemas/id_fields.json` (наше: такой же список, по которому жюри могло бы переименовывать) | `auditenv/idfields.py`, `jury_checks.jury_transform` | `test_id_fields.py` |

| 16 | Может ли политика узнать seed скрытого эпизода? | Через инструменты среды — нет: они не исполняют код, а seed нигде в наблюдениях не встречается (опр. C20). Но генератор публичный: политика, у которой есть репозиторий и интерпретатор, восстанавливает небольшой seed по `episode_id` перебором (seed 73 421 — за 0,96 с). Защита — ключ оценщика `AUDITENV_SECRET` или `AUDITENV_KEY_FILE`: seed превращается в HMAC. По умолчанию ключ выключен, чтобы seed воспроизводил эпизод в любом новом контейнере (опр.: повтор с теми же seed и config) (наше) | `judge.py`: `grader_key`, `effective_seed` | `test_grader_secret.py` |
| 17 | Что жюри считает изоляцией политики по C20, если запускает наш образ без root? | Определения требуют, чтобы в каналах политики не было эталона, seed и ответов оценщика, и обещают смотреть код и изоляцию; как запускается политика жюри, не сказано. Мы измеряем слои на той машине, где идёт проверка, и печатаем их: без root (как в образе с `USER 10001`) отдельного uid нет, остальное действует — пока политика жива, эталона и журнала оценщика на диске нет, процесс среды закрыт для `/proc` и ptrace, форк и запись запрещены. Самая сильная граница — контейнер на политику, `make docker-isolated` (наше) | `auditenv/sandbox.py`, `Dockerfile.policy` | `test_sandbox.py` (попытки побега и положительные контроли) |

## Что мы бы спросили у организаторов

1. Точный текст ответов от 21.11, особенно формат общей фикстуры для теста 12: имена таблиц, полей
   суммы и допуска в эталоне.
2. Совпадает ли наш выбор пар (вопрос 12) с эталонной реализацией жюри, если он где-то отличается
   от правила «по `issue_id`, затем по индексу находки».
3. Есть ли в оценке баллы за презентацию или демонстрацию, кроме пяти критериев раздела 10.
4. Будут ли скрытые эпизоды генерироваться повторно в новом контейнере. Если нет, мы бы включили
   ключ оценщика по умолчанию (вопрос 16).
