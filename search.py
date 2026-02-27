import json
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ====================== ФУНКЦИЯ ЭМБЕДДИНГА ======================
def embed_query(text: str) -> list:
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

# ====================== ПАРСИНГ ЗАПРОСА ЧЕРЕЗ LLM ======================
def parse_query_with_llm(query: str) -> dict:
    """
    Отправляет запрос в LLM и получает структурированные данные:
    {
        "dates": ["14.03", ...],
        "entity_names": ["vehuiah", ...],
        "free_text": "остальной текст для поиска"
    }
    """
    prompt = f"""
Ты — интеллектуальный помощник, который анализирует запрос пользователя и извлекает из него структурированную информацию для поиска в базе данных.

Запрос пользователя: "{query}"

Извлеки из запроса:
1. Все даты в формате ДД.ММ (если есть). Пример: "14 марта" → "14.03", "14.03" → "14.03".
2. Все имена сущностей (если есть). НЕ ПРИДУМЫВАЙ имена, если их нет в запросе. Если в запросе нет конкретных имён, оставь пустой список.
3. Остальной текст для семантического поиска (free_text) — это запрос, из которого удалены все извлечённые даты и имена. Если даты и имена не извлекались, free_text совпадает с исходным запросом.

Ответ выдай строго в формате JSON без пояснений:
{{
    "dates": ["14.03", ...],
    "entity_names": ["vehuiah", ...],
    "free_text": "..."
}}

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
- НИЧЕГО НЕ ПРИДУМЫВАЙ
- ОТВЕЧАЙ КРАТКО
- ПЕРЕПРОВЕРЯЙ СЕБЯ ПЕРЕД ОТВЕТОМ НА СОБЛЮДЕНИЕ ВСЕХ ПРАВИЛ
"""
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:14b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        },
        timeout=300
    )
    resp.raise_for_status()
    answer = resp.json()["response"]
    # Извлечём JSON из ответа (может быть обёрнут в ```json ... ```)
    json_str = answer.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Если не удалось распарсить, вернём пустую структуру
        return {"dates": [], "entity_names": [], "free_text": query}

# ====================== ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА ======================
def generate_answer(prompt: str, context: str) -> str:
    full_prompt = f"Контекст:\n{context}\n\nВопрос: {prompt}\n\nОтвет:"
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:14b",
            "prompt": full_prompt,
            "stream": False
        },
        timeout=600
    )
    resp.raise_for_status()
    return resp.json()["response"]

# ====================== ПОДКЛЮЧЕНИЕ К QDRANT ======================
client = QdrantClient("localhost", port=6333)
COLLECTION_NAME = "legal_docs"

# ====================== ЗАПРОС ПОЛЬЗОВАТЕЛЯ ======================
query = "перечисли имена"
print(f"Запрос: {query}")

# 1. Анализируем запрос
parsed = parse_query_with_llm(query)
dates = parsed.get("dates", [])
entity_names = parsed.get("entity_names", [])
free_text = parsed.get("free_text", "")

print("Распознано:")
print(f"  Даты: {dates}")
print(f"  Имена сущностей: {entity_names}")
print(f"  Текст для поиска: '{free_text}'")

# 2. Строим фильтр
must_conditions = []
if dates:
    # Если несколько дат, используем OR (should)
    # Для одной даты можно must, но для гибкости будем использовать should
    date_conditions = [
        models.FieldCondition(
            key="dates",
            match=models.MatchValue(value=d)
        ) for d in dates
    ]
    must_conditions.append(
        models.Filter(should=date_conditions)  # хотя бы одна из дат
    )
if entity_names:
    name_conditions = [
        models.FieldCondition(
            key="entity_name",
            match=models.MatchValue(value=name)
        ) for name in entity_names
    ]
    must_conditions.append(
        models.Filter(should=name_conditions)
    )

# Объединяем все must-условия (если они есть)
if must_conditions:
    filter_condition = models.Filter(must=must_conditions)
else:
    filter_condition = None

# 3. Определяем текст для семантического поиска
search_text = free_text if free_text.strip() else query
query_vector = embed_query(search_text)

# 4. Выполняем поиск
response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=filter_condition,
    limit=100
)
results = response.points

# 5. Формируем контекст
if not results:
    print("Ничего не найдено.")
    context = "Документы не найдены."
else:
    context_parts = []
    for res in results:
        entity = res.payload.get('entity_name', 'неизвестно')
        dates_list = res.payload.get('dates', [])
        # Добавляем информацию о сущности и её датах
        context_parts.append(
            f"Сущность: {entity}\nДаты: {', '.join(dates_list)}"
        )
    context = "\n\n---\n\n".join(context_parts)

# 6. Генерируем ответ
print("Генерирую ответ...")
answer = generate_answer(query, context)
print(f"Ответ:\n{answer}")