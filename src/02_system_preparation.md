[← Повернутися на головну](../)

---

## 8. Офлайн-етап: підготовка системи до появи пацієнта

### 8.1. Крок 1: Зчитати базу знань "діагноз → симптоми"

**Вхід:** файл, де кожному діагнозу відповідає список симптомів (і, якщо є, частоти/ваги симптомів).

**Результат:** набір записів виду:
* `disease_id`
* `disease_name`
* `symptoms = {s_1, s_2, ...}` (можливо з вагою)

---

### 8.2. Крок 2: Побудувати глобальний словник симптомів

Зібрати всі унікальні симптоми з усієї бази й зафіксувати **єдиний порядок**:

* `symptom_index: symptom → idx`
* `idx → symptom`

**Критично:** однаковий симптом завжди має один і той самий індекс.

---

### 8.3. Крок 3: Векторизувати кожен діагноз у простір симптомів

Для кожного діагнозу будуємо вектор \\(x^{(d)} \in \mathbb{R}^D\\), де \\(D\\) — кількість симптомів у словнику.

**Варіанти заповнення:**

* **бінарний**: \\(x_i = 1\\), якщо симптом присутній у діагнозі, інакше 0
* **ваговий**: \\(x_i = w\\) (частота/вага симптому в діагнозі), інакше 0

**Нормалізація:** L2-нормалізація або інша фіксована схема (щоб "довгі списки симптомів" не домінували).

---

### 8.4. Крок 4: Навчити SOM на векторах діагнозів

**Вхід для SOM:** \\(\{x^{(d)}\}_{d=1..N}\\)

**Налаштування:**
* розмір карти (наприклад 10×10 = 100 нейронів-прототипів)
* метрика відстані (евклідова або косинусна)
* кількість епох, schedule для learning rate і neighborhood radius

**Результат навчання:**
* карта прототипів \\(w_{i,j} \in \mathbb{R}^D\\)
* для кожного діагнозу можна визначити BMU (найближчий нейрон)

---

### 8.5. Крок 5: "Прикріпити" діагнози до нейронів карти

Для кожного діагнозу:
* знайти BMU: \\(\text{BMU}(d) = \arg\min_{i,j}\|x^{(d)} - w_{i,j}\|\\)
* додати діагноз у список цього нейрона

**Результат:** "карта діагнозів" — кожна клітинка має список діагнозів (може бути порожня).

---

### 8.6. Крок 6: Зафіксувати формат SOM-виходу

Вибір формату "SOM-ознаки" для наступних модулів:
* **BMU координати/ID** (жорстко): один нейрон
* **top-k нейронів + їх відстані** (геометрія)
* **top-k нейронів + membership** (відстані, перетворені у ваги та нормовані)

---

## 9. Онлайн-етап: робота з пацієнтом

### 9.1. Крок 1: Отримати первинний опис від пацієнта

**Вхід:** голос/текст + (можливо) файли аналізів

---

### 9.2. Крок 2: Перетворити опис у структурований "case record"

Витягнути:
* симптоми (назви)
* атрибути симптомів (інтенсивність, тривалість, локалізація)
* анамнез (вік, стать, хронічні захворювання)
* виміри/аналізи (числові)

**Результат:** структурований об'єкт "випадок".

---

### 9.3. Крок 3: Перетворити випадок у вектор ознак пацієнта

**Підхід А (SOM працює тільки на симптомах):**
* будуємо \\(x^{(p)}_{sym} \in \mathbb{R}^D\\) у тому ж просторі симптомів, що й діагнози
* додаткові дані (вік, аналізи) підуть потім у Multilabel NN, але НЕ в SOM

---

### 9.4. Крок 4: Спроєктувати пацієнта на SOM (mapping)

Рахуємо відстань від пацієнтського вектора до кожного прототипу:

\\[d_{i,j} = \|x^{(p)}_{sym} - w_{i,j}\|\\]

Далі формуємо SOM-результат: BMU, top-k відстаней, або top-k membership.

---

### 9.5. Крок 5: Витягнути "кандидатний простір" діагнозів

Беремо діагнози з BMU нейрона (мінімально) або з top-k нейронів (краще для диференціалки).

**Результат:**
\\[\mathcal{D}_{cand} = \{d_1, d_2, \dots\}\\]

---

### 9.6. Крок 6: Підготувати вхід для Multilabel NN

Формуємо фінальний вектор для класифікатора:
* **кейс-ознаки** (симптоми + атрибути + анамнез + аналізи)
* **SOM-ознаки** (BMU / top-k distances / top-k membership)
* (опціонально) маска/список кандидатних діагнозів \\(\mathcal{D}_{cand}\\) для обмеження виходу

---

## 10. JSON-схеми даних

### 10.1. `case_record` — структурований випадок пацієнта

```json
{
  "case_id": "string",
  "patient": {
    "patient_id?": "string",
    "age_years?": 0,
    "sex_at_birth?": "male|female|unknown"
  },
  "chief_complaint": {
    "text": "string",
    "onset?": {
      "start_time_iso?": "YYYY-MM-DDThh:mm:ss",
      "duration_hours?": 0,
      "mode?": "sudden|gradual|unknown"
    }
  },
  "symptoms": [
    {
      "name": "string",
      "present": true,
      "severity_0_10?": 0,
      "duration_hours?": 0,
      "location?": "string",
      "notes?": "string"
    }
  ],
  "negated_symptoms?": [
    {
      "name": "string",
      "present": false,
      "notes?": "string"
    }
  ],
  "history?": {
    "conditions?": ["string"],
    "medications?": ["string"],
    "allergies?": ["string"],
    "pregnancy_status?": "pregnant|not_pregnant|unknown"
  },
  "vitals?": {
    "temperature_c?": 0.0,
    "heart_rate_bpm?": 0,
    "resp_rate_bpm?": 0,
    "spo2_percent?": 0,
    "bp_systolic_mmHg?": 0,
    "bp_diastolic_mmHg?": 0
  },
  "labs?": [
    {
      "test": "string",
      "value": 0.0,
      "unit": "string",
      "ref_low?": 0.0,
      "ref_high?": 0.0,
      "datetime_iso?": "YYYY-MM-DDThh:mm:ss"
    }
  ],
  "attachments?": [
    {
      "type": "lab_pdf|image|other",
      "filename": "string",
      "sha256?": "string"
    }
  ],
  "metadata?": {
    "language": "uk",
    "created_at_iso": "YYYY-MM-DDThh:mm:ss",
    "source": "voice|text|mixed"
  }
}
```

**Приклад:**

```json
{
  "case_id": "case_000123",
  "patient": { "age_years": 34, "sex_at_birth": "female" },
  "chief_complaint": {
    "text": "Біль у грудях і задишка",
    "onset": { "duration_hours": 6, "mode": "sudden" }
  },
  "symptoms": [
    { "name": "shortness_of_breath", "present": true, "severity_0_10": 6 },
    { "name": "cough", "present": true, "severity_0_10": 3 },
    { "name": "chest_pain", "present": true, "severity_0_10": 5 }
  ],
  "negated_symptoms": [
    { "name": "diarrhea", "present": false }
  ],
  "vitals": { "temperature_c": 37.4, "spo2_percent": 96, "heart_rate_bpm": 98 },
  "metadata": { "language": "uk", "created_at_iso": "2026-01-06T10:15:00", "source": "voice" }
}
```

---

### 10.2. `x_patient_sym` — вектор симптомів пацієнта у просторі SOM

```json
{
  "case_id": "string",
  "symptom_space": {
    "dictionary_id": "string",
    "dimension": 0
  },
  "encoding": "binary|weighted",
  "vector_sparse": [
    { "idx": 0, "value": 0.0 }
  ],
  "missing_policy?": "unknown_as_0|explicit_missing_flags"
}
```

**Приклад:**

```json
{
  "case_id": "case_000123",
  "symptom_space": { "dictionary_id": "symdict_v1", "dimension": 512 },
  "encoding": "binary",
  "vector_sparse": [
    { "idx": 14, "value": 1 },
    { "idx": 55, "value": 1 },
    { "idx": 201, "value": 1 }
  ],
  "missing_policy": "unknown_as_0"
}
```

---

### 10.3. `som_result` — результат проєкції пацієнта на SOM

```json
{
  "case_id": "string",
  "som_model": {
    "som_id": "string",
    "grid_rows": 0,
    "grid_cols": 0,
    "metric": "euclidean|cosine",
    "prototype_space": "symptom_space"
  },
  "bmu": {
    "row": 0,
    "col": 0,
    "unit_id": "string"
  },
  "top_units": [
    {
      "unit_id": "string",
      "row": 0,
      "col": 0,
      "distance?": 0.0,
      "membership?": 0.0
    }
  ]
}
```

**Приклад:**

```json
{
  "case_id": "case_000123",
  "som_model": { "som_id": "som_10x10_v1", "grid_rows": 10, "grid_cols": 10, "metric": "euclidean", "prototype_space": "symptom_space" },
  "bmu": { "row": 3, "col": 4, "unit_id": "u_3_4" },
  "top_units": [
    { "unit_id": "u_3_4", "row": 3, "col": 4, "distance": 0.18, "membership": 0.64 },
    { "unit_id": "u_3_5", "row": 3, "col": 5, "distance": 0.29, "membership": 0.21 },
    { "unit_id": "u_7_2", "row": 7, "col": 2, "distance": 0.61, "membership": 0.09 }
  ]
}
```

---

### 10.4. `candidate_diagnoses` — кандидати діагнозів із SOM-картки

```json
{
  "case_id": "string",
  "selection_policy": {
    "from_units": "bmu_only|top_k_units",
    "k_units?": 0,
    "max_diagnoses?": 0,
    "dedup": true
  },
  "source_units": [
    { "unit_id": "string", "row": 0, "col": 0, "weight?": 0.0 }
  ],
  "candidates": [
    {
      "disease_id": "string",
      "disease_name": "string",
      "support": {
        "units": [
          { "unit_id": "string", "weight?": 0.0 }
        ]
      }
    }
  ]
}
```

**Приклад:**

```json
{
  "case_id": "case_000123",
  "selection_policy": { "from_units": "top_k_units", "k_units": 3, "max_diagnoses": 50, "dedup": true },
  "source_units": [
    { "unit_id": "u_3_4", "row": 3, "col": 4, "weight": 0.64 },
    { "unit_id": "u_3_5", "row": 3, "col": 5, "weight": 0.21 },
    { "unit_id": "u_7_2", "row": 7, "col": 2, "weight": 0.09 }
  ],
  "candidates": [
    { "disease_id": "D_0102", "disease_name": "Asthma", "support": { "units": [ { "unit_id": "u_3_4", "weight": 0.64 } ] } },
    { "disease_id": "D_0451", "disease_name": "Emphysema", "support": { "units": [ { "unit_id": "u_3_4", "weight": 0.64 } ] } },
    { "disease_id": "D_0770", "disease_name": "Atelectasis", "support": { "units": [ { "unit_id": "u_3_5", "weight": 0.21 } ] } }
  ]
}
```

---

### 10.5. `nn_input_payload` — пакет входів для Multilabel NN

```json
{
  "case_id": "string",
  "model_target": {
    "nn_model_id": "string",
    "label_space_id": "string"
  },
  "features": {
    "clinical": {
      "age_years?": 0,
      "sex_at_birth?": "male|female|unknown",
      "vitals?": { "temperature_c?": 0.0, "spo2_percent?": 0 },
      "labs_numeric?": [
        { "test_id": "string", "value": 0.0 }
      ],
      "symptoms_vector": {
        "dictionary_id": "string",
        "dimension": 0,
        "vector_sparse": [ { "idx": 0, "value": 0.0 } ]
      }
    },
    "som": {
      "som_id": "string",
      "bmu_unit_id": "string",
      "top_units": [
        { "unit_id": "string", "distance?": 0.0, "membership?": 0.0 }
      ]
    }
  },
  "constraints?": {
    "candidate_diseases?": [ "string" ],
    "max_outputs?": 0
  }
}
```

**Приклад:**

```json
{
  "case_id": "case_000123",
  "model_target": { "nn_model_id": "multilabel_v1", "label_space_id": "disease_labels_v1" },
  "features": {
    "clinical": {
      "age_years": 34,
      "sex_at_birth": "female",
      "vitals": { "temperature_c": 37.4, "spo2_percent": 96 },
      "symptoms_vector": {
        "dictionary_id": "symdict_v1",
        "dimension": 512,
        "vector_sparse": [
          { "idx": 14, "value": 1 },
          { "idx": 55, "value": 1 },
          { "idx": 201, "value": 1 }
        ]
      }
    },
    "som": {
      "som_id": "som_10x10_v1",
      "bmu_unit_id": "u_3_4",
      "top_units": [
        { "unit_id": "u_3_4", "membership": 0.64 },
        { "unit_id": "u_3_5", "membership": 0.21 },
        { "unit_id": "u_7_2", "membership": 0.09 }
      ]
    }
  },
  "constraints": {
    "candidate_diseases": [ "D_0102", "D_0451", "D_0770" ],
    "max_outputs": 20
  }
}
```

---

## 11. Candidate Selector — формальний опис

### 11.1. Формальне призначення

**Candidate Selector** — це детермінований алгоритм, який на основі результату SOM формує **скінченну множину діагнозів-кандидатів** для подальшої диференціальної діагностики.

**Формально:**

\\[\text{CandidateSelector} : (\text{som\_result}, \text{som\_index}) \longrightarrow \mathcal{D}_{cand}\\]

де:
* `som_result` — результат проєкції пацієнта на SOM
* `som_index` — зафіксована мапа «нейрон → діагнози»
* \\(\mathcal{D}_{cand}\\) — множина діагнозів-кандидатів

---

### 11.2. Політика відбору з параметрами \\((k, \tau, \alpha)\\)

**Параметри:**
* \\(k\\) — максимальна кількість юнітів
* \\(\tau\\) — мінімальний поріг membership
* \\(\alpha\\) — цільова cumulative mass

---

### 11.3. Алгоритм Candidate Selector (по кроках)

**Крок 1. Вибір релевантних нейронів**

Беремо всі нейрони \\(u\\), такі що:

\\[u \in \text{top\_units}, \quad u.\text{rank} \le k, \quad u.\text{membership} \ge \tau\\]

**Крок 2. Агрегація діагнозів**

Формуємо початкову множину:

\\[\mathcal{D}_0 = \bigcup_{u} \text{som\_index}[u]\\]

**Крок 3. Обчислення підтримки діагнозів**

Для кожного діагнозу \\(d\\) обчислюємо **підтримку**:

\\[\text{support}(d) = \sum_{u \ni d} \text{membership}(u)\\]

**Крок 4. Сортування і обрізання**

Сортуємо за \\(\text{support}(d)\\) за спаданням, обрізаємо до `max_diagnoses`.

---

### 11.4. Ключові інваріанти

Candidate Selector **гарантує**:

1. \\(\mathcal{D}_{cand}\\) скінченна
2. Кожен кандидат має пояснення (через які нейрони)
3. Немає діагнозів «поза SOM»
4. Розмір контрольований
5. Немає втрати альтернатив (якщо параметри \\(k, \tau, \alpha\\) налаштовані правильно)

---

## 12. Побудова мапи "нейрон → діагнози"

### 12.1. Формальне математичне означення

Нехай:
* \\(U\\) — множина нейронів SOM: \\(U = \{u_{1}, u_{2}, \dots, u_{100}\}\\)
* \\(\mathcal{D}\\) — множина всіх діагнозів

Тоді **мапа**:

\\[I : U \rightarrow 2^{\mathcal{D}}\\]

де:
* \\(2^{\mathcal{D}}\\) — множина всіх підмножин діагнозів
* \\(I(u)\\) — **множина діагнозів**, прикріплених до нейрона \\(u\\)

---

### 12.2. Алгоритм побудови

Після навчання SOM:

1. Беремо кожен діагноз \\(d\\)
2. Маємо його вектор \\(x^{(d)}\\)
3. Знаходимо BMU:
   \\[u^{*}(d) = \arg\min_{u \in U} \|x^{(d)} - w_u\|\\]
4. Додаємо діагноз у мапу:
   \\[I[u^{*}(d)] \leftarrow I[u^{*}(d)] \cup \{d\}\\]

Після цього **мапа фіксується** (read-only).

---

### 12.3. Приклад структури даних

```json
{
  "u_3_4": ["Asthma", "Emphysema", "Pulmonary Eosinophilia"],
  "u_3_5": ["Atelectasis", "Lung Contusion"],
  "u_7_2": ["Panic Disorder", "Anxiety Disorder"],
  "u_1_9": []
}
```

---

## 13. Формальні означення базових сутностей

### 13.1. Клінічний випадок \\((c)\\)

**Визначення:**

> **Клінічний випадок \\((c)\\)** — це формалізований опис *одного конкретного звернення пацієнта*, який містить усю доступну на даний момент медичну інформацію, але **не містить діагнозу**.

**Формально:**

\\[c = (\text{symptoms},\ \text{clinical\_data},\ \text{context})\\]

---

### 13.2. Простір діагнозів \\(\mathcal{D}\\)

> \\(\mathcal{D}\\) — це фіксована множина всіх діагнозів, які система взагалі вміє розпізнавати.

\\[\mathcal{D} = \{\text{Asthma},\ \text{Emphysema},\ \text{Atelectasis},\ \dots\}\\]

Ця множина формується **офлайн** і є label space для Multilabel NN.

---

### 13.3. Multilabel NN — формально

**Визначення:**

> Multilabel NN — це модель, яка для **одного клінічного випадку** \\(c\\) **одночасно** оцінює *можливість кожного діагнозу* з \\(\mathcal{D}\\).

**Формально:**

\\[f_\theta : \mathcal{C} \longrightarrow [0,1]^{|\mathcal{D}|}\\]

де:
* \\(\mathcal{C}\\) — множина всіх можливих клінічних випадків
* \\(f_\theta(c)_d \in [0,1]\\) — **ступінь впевненості**, що діагноз \\(d\\) сумісний з випадком \\(c\\)

---

### 13.4. Вхідний вектор Multilabel NN

Для кожного клінічного випадку \\(c\\) будується вектор ознак:

\\[x(c) = [x_{clinical},\ x_{symptoms},\ x_{som}]\\]

де:
* \\(x_{symptoms} \in \mathbb{R}^D\\) — вектор симптомів
* \\(x_{clinical} \in \mathbb{R}^C\\) — несимптомні клінічні ознаки (вік, стать, vitals, labs)
* \\(x_{som} = (m_1, m_2, \dots, m_K)\\) — SOM-контекст (membership top-K юнітів)

---

### 13.5. Вихід Multilabel NN

\\[\hat{y}(c) = (\hat{y}_{d_1}, \hat{y}_{d_2}, \dots, \hat{y}_{d_L})\\]

де:
* кожен елемент відповідає **одному діагнозу**
* всі виходи незалежні (sigmoid)

Результат — **ранжований набір гіпотез**, а не фінальний діагноз.

---

