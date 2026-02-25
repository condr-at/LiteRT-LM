# Аудит litert-lm-fork: обновлённая версия (после верификации)

**Дата:** 24 февраля 2026  
**Статус:** ✅ Верифицировано по кодовой базе

---

## Общая оценка

**7/10** — код нормального качества для внутреннего проекта. Архитектура осмысленная, абстракции чистые, стиль консистентный с upstream.

**Что изменилось после верификации:**
- Все критические и серьёзные проблемы подтверждены
- Ложные тревоги (race conditions, memory leaks, buffer leaks) опровергнуты
- Добавлены точные номера строк и цитаты кода
- Уточнены формулировки проблем

---

## Подтверждённые реальные проблемы

### 🔴 КРИТИЧЕСКИЕ

#### 1. Callback вызывается под мьютексом → deadlock risk
**Файл:** `runtime/framework/resource_management/execution_manager.cc`

**Строки 304-307:**
```cpp
} else if (IsTaskEndState(task_state)) {
  // Only emit immediate callback when task starts already in terminal state
  // due to failed/cancelled dependencies. For kCreated, defer callback until
  // queue/start notifications to avoid re-entrant callback hazards.
  task_lookup_.at(task_id).callback(Responses(task_state));  // ← ВЫЗОВ ПОД МЮТЕКСОМ
}
```

**Строки 327-328, 335-336 (QueueTask):**
```cpp
if (task_lookup_.at(task_id).callback != nullptr) {
  task_lookup_.at(task_id).callback(error_status);  // ← ТОЖЕ ПОД МЮТЕКСОМ
}
```

**Почему это проблема:**
- Код вызывается под `session_and_task_lookup_mutex_`
- Если callback обратится к любому методу `ExecutionManager` (`Cancel()`, `GetSessionInfo()`, `AddTask()`), который требует тот же мьютекс — мгновенный deadlock
- Сам код комментирует риск (строки 304-306: "to avoid re-entrant callback hazards"), но всё равно так делает

**Фикс:**
```cpp
// Захватить callback и response в локальные переменные ДО release мьютекса
auto callback = std::move(task_lookup_.at(task_id).callback);
auto response = Responses(task_state);
// ... release mutex ...
callback(response);  // ← Вызывать ПОСЛЕ
```

**Где исправить:**
- `CreateTask()` — строки 304-307
- `QueueTask()` — строки 327-328, 335-336

---

#### 2. При ошибке CloneAsync callback теряется
**Файл:** `runtime/framework/resource_management/execution_manager.cc:945-950`

```cpp
auto task_info = StartTask(task_id);
if (!task_info.ok()) {
  FinishTaskAndLogErrors(task_id, task_info.status(),
                         [](absl::StatusOr<Responses> responses) {});  // ← ПУСТОЙ CALLBACK
  return;
}
```

**Почему это проблема:**
- Оригинальный callback заменяется пустой лямбдой
- Kotlin-код никогда не узнает о провале клонирования
- Задача остаётся в подвешенном состоянии с точки зрения caller'а

**Сравнение с правильным паттерном (AddDecodeTask, строки 784-789):**
```cpp
auto task_info = StartTask(task_id);
if (!task_info.ok()) {
  FinishTaskAndLogErrors(task_id, task_info.status(),
                         std::move(callback));  // ← Правильно: оригинальный callback
  return;
}
```

**Фикс:** Захватить оригинальный callback в замыкание task и вызывать его при ошибке `StartTask()` по аналогии с `AddDecodeTask`.

---

### 🟠 СЕРЬЁЗНЫЕ

#### 3. Silent failures в C API
**Файл:** `c/engine.cc:165-170, 337-342`

```cpp
void litert_lm_session_config_set_max_output_tokens(
    LiteRtLmSessionConfig* config, int max_output_tokens) {
  if (config && config->config) {
    config->config->SetMaxOutputTokens(max_output_tokens);
  }  // ← молча игнорирует nullptr, caller не знает об ошибке
}

void litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings) {
  if (settings && settings->settings) {
    settings->settings->GetMutableBenchmarkParams();  // ← результат игнорируется
  }
}
```

**Почему это проблема:**
- Многие C-функции возвращают `void`
- Caller не получает никакого сигнала об ошибке
- Ошибки проглатываются silently

**Фикс:** Изменить сигнатуры на возвращающие `LiteRtStatus` или добавить callback для ошибок.

---

#### 4. Conversation::Clone глотает Unimplemented от CloneState()
**Файл:** `runtime/conversation/conversation.cc:546-549`

```cpp
auto status = model_data_processor->CloneState(*model_data_processor_);
if (!status.ok() && !absl::IsUnimplemented(status)) {
  return status;
}  // ← Unimplemented игнорируется без логирования
```

**Почему это проблема:**
- Клон создаётся в деградированном состоянии (без состояния `ModelDataProcessor`)
- Caller не получает никакого уведомления о деградации
- Отладка таких ситуаций крайне затруднена

**Минимальный фикс:**
```cpp
auto status = model_data_processor->CloneState(*model_data_processor_);
if (!status.ok() && !absl::IsUnimplemented(status)) {
  return status;
}
if (absl::IsUnimplemented(status)) {
  ABSL_LOG(WARNING) << "Conversation::Clone: CloneState unimplemented, "
                    << "cloning in degraded state (no ModelDataProcessor state)";
}
```

---

#### 5. ExecutionManager — один мьютекс на всё
**Файл:** `runtime/framework/resource_management/execution_manager.h`

```cpp
absl::Mutex session_and_task_lookup_mutex_;
absl::flat_hash_map<SessionId, std::shared_ptr<SessionInfo>> session_lookup_
    ABSL_GUARDED_BY(session_and_task_lookup_mutex_);
absl::flat_hash_map<TaskId, TaskInfo> task_lookup_
    ABSL_GUARDED_BY(session_and_task_lookup_mutex_);
```

**Почему это проблема:**
- `session_and_task_lookup_mutex_` защищает и сессии, и таски одновременно
- При многосессионной нагрузке возникает contention
- Все операции (создание сессии, создание таски, обновление статуса, callback) конкурируют за один мьютекс

**Фикс:** Разделить мьютексы:
- `session_mutex_` для `session_lookup_`
- `task_mutex_` для `task_lookup_`

---

#### 6. Инвариант single-thread execution не задокументирован
**Файл:** `runtime/framework/resource_management/execution_manager.h:258-265`

```cpp
ExecutionManager(...)
    : tokenizer_(std::move(tokenizer)),
      resource_manager_(std::move(resource_manager)),
      litert_env_(litert_env) {
  execution_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                   /*max_num_threads=*/1);  // ← НИГДЕ НЕ ЗАДОКУМЕНТИРОВАНО
  callback_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"callback_thread_pool",
                                   /*max_num_threads=*/1);
}
```

**Файл:** `runtime/framework/resource_management/resource_manager.cc:590-602`

```cpp
MovableMutexLock lock(&executor_mutex_);
if (current_handler_ != llm_context_handler) {
  return absl::InternalError(
      "CLONE_RUNTIME_STATE_SOURCE_INVALID: context handler has no runtime "
      "config/state and is not current_handler_. Refusing to clone with "
      "executor state from a different active handler.");
}
ASSIGN_OR_RETURN(runtime_config, llm_executor_->GetRuntimeConfig());
ASSIGN_OR_RETURN(runtime_state, llm_executor_->GetRuntimeState());
```

**Почему это проблема:**
- `CloneContextHandler()` читает runtime state из executor без гарантии стабильности источника
- Корректность зависит от того, что `execution_thread_pool` имеет `max_num_threads=1`
- clone и decode выполняются последовательно — это критический инвариант
- Если кто-то поднимет parallelism, всё сломается тонкими гонками

**Фикс:**
1. Добавить комментарий в конструктор `ExecutionManager` о single-thread инварианте
2. Добавить `RET_CHECK(max_num_threads == 1)` в `ExecutionManager::Create()`
3. Задокументировать в `CloneContextHandler()` зависимость от single-thread execution

---

#### 7. Ноль автотестов на кастомный функционал

**Найдено тестов:**
- `runtime/conversation/conversation_test.cc:1414` — `Clone()` тест (поверхностный)
- `runtime/conversation/model_data_processor/gemma3_data_processor_test.cc:1200` — `CloneState` тест
- ~30 тестов на обычную работу сессий (`session_advanced_test.cc`, `session_basic_test.cc`)

**Не покрыто тестами:**
- ❌ Clone() → независимость KV-кешей оригинала и клона
- ❌ CloneAsync() с зависимостями
- ❌ Переключение context handlers
- ❌ Отмена clone task
- ❌ RestoreContext() с валидацией имён буферов
- ❌ Context switching между сессиями

**Единственное свидетельство работы:** ручной лог `clone35_regression_log.txt` (425K строк) с телефона.

**Фикс:** Добавить тесты в `session_advanced_test.cc` и `resource_manager_test.cc`.

---

### 🟡 НЕЗНАЧИТЕЛЬНЫЕ

#### 8. session_advanced.h:96 — деструктор глотает DEADLINE_EXCEEDED
```cpp
~SessionAdvanced() override { WaitUntilDone().IgnoreError(); };
```
**Проблема:** Ошибки таймаута игнорируются, возможна утечка ресурсов.

---

#### 9. threadpool.cc:167-169 — bare unlock/lock (exception-unsafe)
```cpp
mutex_.unlock();
std::move(task_to_run)();  // ← если выбросит exception, мьютекс не залочится обратно
mutex_.lock();
```
**Фикс:** Использовать RAII-обёртку или `absl::ReleasableMutexLock`.

---

#### 10. tasks.cc:385 — ошибка инициализации sampler игнорируется
```cpp
compiled_model_executor->InitializeSampler().IgnoreError();
```
**Проблема:** Ошибка инициализации sampler может привести к некорректной генерации.

---

#### 11. session_basic.cc:68-71 — статический указатель никогда не удаляется
```cpp
absl::flat_hash_set<LlmExecutor*>* SessionBasic::occupied_executors_ =
    new absl::flat_hash_set<LlmExecutor*>();
ABSL_CONST_INIT absl::Mutex SessionBasic::occupied_executors_mu_(absl::kConstInit);
```
**Проблема:** Memory leak при завершении программы (не критично для long-running процессов).
**Фикс:** Использовать function-local static или `LeakySingleton`.

---

#### 12. llm_litert_compiled_model_executor.cc:1585-1602 — нет валидации имён KV-буферов при restore
**Проблема:** При `RestoreContext()` имена буферов не валидируются на соответствие ожидаемым. Может привести к тихой порче состояния при несовпадении имён.

---

## Опровержения (что агенты неверно описали)

| Утверждение | Реальность | Доказательство |
|---|---|---|
| «Race condition на `last_task_ids_`» | ❌ При single-thread execution pool гонки нет | `execution_manager.h:259-261`, `max_num_threads=1` |
| «Dangling pointers в `handlers_`» | ❌ Ложная тревога. `handlers_` — raw pointers, но защищены мьютексом, деструктор `ContextHandler` явно вызывает `RemoveHandler(this)` | `context_handler.cc:83-86` |
| «Memory leaks в C API» | ❌ Неверно. Внутри используются `std::unique_ptr` — утечек нет. Реальная проблема — silent failures | `c/engine.cc` |
| «Deadlock risk от одного мьютекса» | ❌ При одном мьютексе дедлок по кольцевому ожиданию невозможен. Реальная проблема — contention. Реальный дедлок — от callback под локом (проблема #1) | `execution_manager.h` |
| «Buffer leak при partial CloneContext()» | ❌ Ложная тревога. `cloned_kv_cache_buffers` — локальная переменная на стеке. При раннем return деструктор `flat_hash_map` освобождает все TensorBuffer через RAII | `llm_litert_compiled_model_executor.cc:1543-1548` |
| «Migrate Bazel → CMake» | ❌ Абсурд для проекта, рождённого в Bazel-экосистеме Google | — |
| «1000+ concurrent clone stress tests» | ❌ При single-thread execution pool конкурентность клонирования невозможна по конструкции | — |

---

## Противоречия между агентами (разрешены)

| Тема | Агент 1 (первый аудит) | Агент 2 (проверка) | Агент 3 (финальный) | Реальность |
|---|---|---|---|---|
| Оценка кода | 6/10 | «ближе к 7/10» | 7/10 | **7/10** ✅ |
| CloneAsync race | Критическая | Преувеличено, есть lock | Нет гонки при single-thread; реальный deadlock — callback под мьютексом | **Callback под мьютексом** ✅ |
| Buffer leak | Не упоминает | Не упоминает | Сначала нашёл, потом опроверг сам себя — RAII работает | **RAII работает** ✅ |
| C API утечки | Да | Нет, unique_ptr | Нет, но silent failures есть | **Silent failures** ✅ |
| Deadlock risk | Да, от lock ordering | «Невозможен при одном mutex» | Реальный deadlock есть, но другой — callback под локом | **Callback под локом** ✅ |

---

## План исправлений (по приоритету)

### PR 1 — Defer callback outside mutex
**Файлы:** `runtime/framework/resource_management/execution_manager.cc`  
**Изменения:**
- `CreateTask()` — строки 304-307
- `QueueTask()` — строки 327-328, 335-336

**Код:**
```cpp
// Захватить callback и response в локальные переменные до release мьютекса
absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback_to_invoke;
absl::StatusOr<Responses> response_to_invoke;
{
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  // ... existing logic ...
  if (IsTaskEndState(task_state)) {
    callback_to_invoke = std::move(task_lookup_.at(task_id).callback);
    response_to_invoke = Responses(task_state);
  }
}
// Вызывать вне мьютекса
if (callback_to_invoke != nullptr) {
  callback_to_invoke(response_to_invoke);
}
```

---

### PR 2 — Fix callback loss on clone failure
**Файл:** `runtime/framework/resource_management/execution_manager.cc:941-950`  
**Изменения:** По аналогии с `AddDecodeTask` (строки 784-789)

**Код:**
```cpp
auto task = [this, task_id, session_id, cloned_session_id, callback = std::move(callback)]() mutable -> void {
  auto task_info = StartTask(task_id);
  if (!task_info.ok()) {
    FinishTaskAndLogErrors(task_id, task_info.status(), std::move(callback));  // ← Исправлено
    return;
  }
  // ... rest of code ...
};
```

---

### PR 3 — Add degradation warning in Conversation::Clone()
**Файл:** `runtime/conversation/conversation.cc:546-549`  
**Изменения:** Добавить `ABSL_LOG(WARNING)` при `IsUnimplemented(status)`

**Код:**
```cpp
auto status = model_data_processor->CloneState(*model_data_processor_);
if (!status.ok() && !absl::IsUnimplemented(status)) {
  return status;
}
if (absl::IsUnimplemented(status)) {  // ← Добавить
  ABSL_LOG(WARNING) << "Conversation::Clone: CloneState unimplemented, "
                    << "cloning in degraded state (no ModelDataProcessor state)";
}
```

---

### PR 4 — Document single-thread invariant + RET_CHECK
**Файлы:**
- `runtime/framework/resource_management/execution_manager.h:258-265` — комментарий
- `runtime/framework/resource_management/execution_manager.cc:697-713` — `RET_CHECK`
- `runtime/framework/resource_management/resource_manager.cc:590-602` — комментарий о зависимости

**Код:**
```cpp
// В конструкторе ExecutionManager:
ExecutionManager(...) {
  // Single-thread execution invariant: clone and decode tasks execute
  // sequentially to ensure stable runtime state during CloneContextHandler().
  // Violating this invariant causes subtle race conditions.
  execution_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                   /*max_num_threads=*/1);
  // ...
}

// В ExecutionManager::Create():
absl::StatusOr<std::unique_ptr<ExecutionManager>> ExecutionManager::Create(...) {
  // ...
  RET_CHECK(resource_manager->GetLlmExecutor()->GetMaxNumThreads() == 1)
      << "LLM executor must have max_num_threads=1 for correct clone semantics";
  // ...
}
```

---

### PR 5 — Automated clone/restore tests
**Файлы:**
- `runtime/core/session_advanced_test.cc` — новые тесты
- `runtime/conversation/conversation_test.cc` — расширение существующих

**Тесты:**
1. `CloneIndependentKVCache` — клон имеет независимый KV-кеш
2. `CloneAsyncWithDependencies` — clone task с зависимостями
3. `ContextHandlerSwitching` — переключение handlers
4. `CancelCloneTask` — отмена clone task
5. `RestoreContextBufferValidation` — валидация буферов при restore

---

## Что НЕ нужно делать

| Действие | Причина |
|---|---|
| C API RAII wrappers | Не используем C API напрямую (путь: Kotlin → JNI → C++ Session) |
| Заменять raw pointers в `handlers_` | Мьютекс + деструктор достаточны, ложная тревога |
| Формализовать lock-order документ | При single-thread execution неактуально |
| Migrate Bazel → CMake | Абсурдно для проекта Google, рождённого в Bazel |
| Concurrent clone stress tests | При single-thread pool конкурентность невозможна |

---

## Приоритизация

| Приоритет | PR | Усилие | Риск |
|---|---|---|---|
| 🔴 P0 | PR 1 — Defer callback | 1-2 часа | Высокий (deadlock) |
| 🔴 P0 | PR 2 — Fix callback loss | 30 мин | Высокий (silent failure) |
| 🟠 P1 | PR 3 — Add warning | 15 мин | Средний (debuggability) |
| 🟠 P1 | PR 4 — Document invariant | 1 час | Средний (future-proofing) |
| 🟡 P2 | PR 5 — Tests | 1-2 дня | Низкий (quality) |

---

## Резюме

Аудит подтверждён. Все критические и серьёзные проблемы реальны. Ложные тревоги опровергнуты. План исправлений корректен и приоритизирован правильно.

**Следующие шаги:**
1. Создать PR 1 (defer callback) — критично
2. Создать PR 2 (fix callback loss) — критично
3. Создать PR 3-4 (warning + документация) — важно
4. Создать PR 5 (тесты) — важно для качества
