import json
import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ====================== ПАРСИНГ ЗАПРОСА ЧЕРЕЗ LLM ======================
def parse_query_with_llm(query: str) -> dict:
    """
    Отправляет запрос в LLM и получает структурированные данные:
    {
        "dates": ["14.03", ...],
        "entity_names": ["vehuiah", ...],
        "free_text": "..."
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

# ====================== ФУНКЦИИ ФИЛЬТРАЦИИ В QDRANT ======================
def build_filter(dates: list, entity_names: list) -> models.Filter:
    """
    Строит фильтр Qdrant на основе списков дат и имён сущностей.
    Логика: (dates[0] OR dates[1] OR ...) AND (entity_names[0] OR entity_names[1] OR ...)
    Если один из списков пуст, соответствующая часть фильтра не добавляется.
    """
    must_conditions = []

    if dates:
        date_conditions = [
            models.FieldCondition(
                key="dates",
                match=models.MatchValue(value=d)
            ) for d in dates
        ]
        must_conditions.append(models.Filter(should=date_conditions))

    if entity_names:
        name_conditions = [
            models.FieldCondition(
                key="entity_name",
                match=models.MatchValue(value=name)
            ) for name in entity_names
        ]
        must_conditions.append(models.Filter(should=name_conditions))

    if must_conditions:
        return models.Filter(must=must_conditions)
    else:
        return None  # нет условий – вернуть все точки

def fetch_chunks(collection_name: str, filter_condition: models.Filter, limit: int = 1000):
    """
    Получает все точки (чанки), соответствующие фильтру, используя scroll.
    Параметр limit определяет размер одной страницы.
    Возвращает список точек.
    """
    client = QdrantClient("localhost", port=6333)
    all_points = []
    offset = None
    page = 1

    while True:
        print(f"    Загрузка страницы {page}...")
        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        points = result[0]
        all_points.extend(points)
        print(f"      Получено {len(points)} чанков на странице")
        offset = result[1]
        if offset is None:
            break
        page += 1

    return all_points

# ====================== ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА ======================
def generate_answer(prompt: str, context: str) -> str:
    full_prompt = f"""Ты — помощник, который отвечает на вопросы пользователя, используя только предоставленный контекст. 
Отвечай максимально кратко, без лишних пояснений, комментариев или оценок. 
Если в контексте есть информация, просто перечисли её в ответ на вопрос. 
Если информация отсутствует, скажи "Информация не найдена".

Контекст:
{context}

Вопрос пользователя: {prompt}

Ответ:"""
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

# ====================== ОСНОВНАЯ ЛОГИКА ======================
if __name__ == "__main__":
    COLLECTION_NAME = "legal_docs"

    query = "перечисли все имена ангелов, которые тебе известны"
    print(f"Запрос: {query}\n")
    print("=" * 60)

    # 1. Анализируем запрос через LLM
    print("Этап 1: Парсинг запроса LLM...")
    t1 = time.perf_counter()
    parsed = parse_query_with_llm(query)
    t2 = time.perf_counter()
    dates = parsed.get("dates", [])
    entity_names = parsed.get("entity_names", [])
    free_text = parsed.get("free_text", "")
    print(f"  Распознано: даты={dates}, имена={entity_names}, текст='{free_text}'")
    print(f"  Время: {t2 - t1:.2f} сек\n")

    # 2. Строим фильтр
    print("Этап 2: Построение фильтра...")
    t1 = time.perf_counter()
    filter_condition = build_filter(dates, entity_names)
    t2 = time.perf_counter()
    print(f"  Фильтр построен (условия: {dates} и {entity_names})")
    print(f"  Время: {t2 - t1:.4f} сек\n")

    # 3. Получаем чанки по фильтру
    print("Этап 3: Загрузка чанков из Qdrant...")
    t1 = time.perf_counter()
    results = fetch_chunks(COLLECTION_NAME, filter_condition, limit=100)
    t2 = time.perf_counter()
    print(f"  Всего получено чанков: {len(results)}")
    print(f"  Время: {t2 - t1:.2f} сек\n")

    # 4. Формируем контекст для LLM (теперь с учётом наличия имён в запросе)
    print("Этап 4: Формирование контекста для LLM...")
    t1 = time.perf_counter()

    if not results:
        context = "Документы не найдены."
    else:
        # Если в запросе не было конкретных имён, выводим только уникальные имена (без дат)
        if not entity_names:
            # Собираем уникальные имена сущностей
            names_found = sorted(set(res.payload.get('entity_name', 'неизвестно') for res in results))
            context = "Найденные имена сущностей:\n" + "\n".join(f"- {name}" for name in names_found)
        else:
            # Иначе (есть конкретные имена) выводим полную информацию: имя + даты
            context_lines = []
            for res in results:
                entity = res.payload.get('entity_name', 'неизвестно')
                dates_list = res.payload.get('dates', [])
                dates_str = ', '.join(dates_list) if dates_list else 'нет дат'
                context_lines.append(f"Сущность: {entity}, даты: {dates_str}")
            context = "\n".join(context_lines)

    t2 = time.perf_counter()
    print(f"  Контекст сформирован (длина: {len(context)} символов)")
    print(f"  Время: {t2 - t1:.2f} сек\n")
    print("  Содержимое контекста:")
    print(context)

    # 5. Генерируем ответ
    print("Этап 5: Генерация ответа LLM...")
    t1 = time.perf_counter()
    answer = generate_answer(query, context)
    t2 = time.perf_counter()
    print(f"  Время генерации ответа: {t2 - t1:.2f} сек\n")
    print("=" * 60)
    print("ОТВЕТ:")
    print(answer)