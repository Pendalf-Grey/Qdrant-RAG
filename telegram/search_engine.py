import json
import logging
import re
from typing import List, Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ====================== НАСТРОЙКИ ======================
COLLECTION_NAME = "legal_docs"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:14b"

# ====================== ЛОГИРОВАНИЕ ======================
logger = logging.getLogger(__name__)

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================

def _is_count_query(query: str) -> bool:
    """
    Определяет, является ли запрос вопросом о количестве (сколько, количество и т.п.)
    """
    # Приводим к нижнему регистру для удобства
    q = query.lower().strip()
    # Список триггерных слов и фраз
    triggers = [
        r'\bсколько\b',
        r'\bколичество\b',
        r'\bчисло\b',
        r'\bсколько всего\b',
        r'\bкакое количество\b',
        r'\bскока\b',          # разговорный вариант
        r'\bкол-во\b',
        r'\bцифр[уа]?\b',      # "напиши цифру", "цифра"
    ]
    for pattern in triggers:
        if re.search(pattern, q):
            return True
    return False

# ====================== СИНХРОННЫЕ ФУНКЦИИ (ВНУТРЕННИЕ) ======================

def _parse_query_with_llm(query: str) -> dict:
    """
    Синхронная функция: отправляет запрос в LLM и получает структурированные данные.
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
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        },
        timeout=300
    )
    resp.raise_for_status()
    answer = resp.json()["response"]
    json_str = answer.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Не удалось распарсить JSON из ответа LLM. Ответ: {answer}")
        return {"dates": [], "entity_names": [], "free_text": query}


def _build_filter(dates: List[str], entity_names: List[str]) -> Optional[models.Filter]:
    """
    Строит фильтр Qdrant на основе списков дат и имён сущностей.
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
    return None


def _fetch_chunks(filter_condition: Optional[models.Filter], limit: int = 1000) -> List[dict]:
    """
    Синхронная функция: получает все чанки из Qdrant, соответствующие фильтру.
    Возвращает список словарей с полями entity_name и dates.
    """
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    all_points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        points = result[0]
        all_points.extend(points)
        offset = result[1]
        if offset is None:
            break
    # Извлекаем только нужные поля из payload
    return [
        {
            "entity_name": p.payload.get("entity_name", "неизвестно"),
            "dates": p.payload.get("dates", [])
        }
        for p in all_points
    ]


def _generate_answer(prompt: str, context: str) -> str:
    """
    Синхронная функция: генерирует ответ на основе контекста.
    Улучшенный промпт с требованием перечислить все имена без пропусков.
    """
    full_prompt = f"""Ты — помощник, который отвечает на вопросы пользователя, используя только предоставленный контекст. Отвечай максимально кратко, без лишних пояснений, комментариев или оценок. Если в контексте есть информация, просто перечисли её в ответ на вопрос. Если в контексте представлен список имён, перечисли все имена из контекста, ничего не пропуская. Если информации недостаточно, скажи "Информация не найдена".

Контекст:
{context}

Вопрос пользователя: {prompt}

Ответ:"""
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": full_prompt,
            "stream": False
        },
        timeout=600
    )
    resp.raise_for_status()
    return resp.json()["response"]


# ====================== АСИНХРОННАЯ ФУНКЦИЯ ДЛЯ ВНЕШНЕГО ИСПОЛЬЗОВАНИЯ ======================

async def process_query(query: str) -> str:
    """
    Асинхронная функция, запускающая все этапы обработки запроса в потоках.
    Возвращает итоговый ответ (строку).
    """
    import asyncio

    logger.info(f"Начало обработки запроса: {query}")

    # Определяем, является ли запрос вопросом о количестве (без вызова LLM)
    is_count = _is_count_query(query)

    # 1. Парсинг запроса (всегда нужен для извлечения фильтров, даже для count-запросов)
    parsed = await asyncio.to_thread(_parse_query_with_llm, query)
    dates = parsed.get("dates", [])
    entity_names = parsed.get("entity_names", [])
    logger.info(f"Распознано: даты={dates}, имена={entity_names}")

    # 2. Построение фильтра
    filter_condition = _build_filter(dates, entity_names)

    # 3. Получение чанков из Qdrant
    chunks = await asyncio.to_thread(_fetch_chunks, filter_condition, 100)

    # Если это вопрос о количестве, возвращаем подсчитанное число без вызова LLM
    if is_count:
        if not chunks:
            count = 0
        else:
            # Собираем уникальные имена сущностей
            unique_names = set(chunk["entity_name"] for chunk in chunks)
            count = len(unique_names)
        # Формируем ответ в зависимости от наличия фильтров
        if dates or entity_names:
            # Были фильтры, уточняем в ответе
            filters_desc = []
            if dates:
                filters_desc.append(f"дата(ы): {', '.join(dates)}")
            if entity_names:
                filters_desc.append(f"имя(ена): {', '.join(entity_names)}")
            filter_str = " с " + " и ".join(filters_desc)
            answer = f"Найдено {count} уникальных сущностей{filter_str}."
        else:
            answer = f"Всего известно {count} ангелов."
        logger.info(f"Ответ на count-запрос: {answer}")
        return answer

    # Иначе продолжаем обычный путь с формированием контекста и генерацией ответа

    # 4. Формирование контекста (с учётом наличия имён в запросе)
    if not chunks:
        context = "Документы не найдены."
    else:
        # Если в запросе не было конкретных имён, выводим только уникальные имена (без дат)
        if not entity_names:
            # Собираем уникальные имена сущностей
            names_found = sorted(set(chunk["entity_name"] for chunk in chunks))
            context = "Найденные имена сущностей:\n" + "\n".join(f"- {name}" for name in names_found)
        else:
            # Иначе (есть конкретные имена) выводим полную информацию: имя + даты
            context_lines = []
            for chunk in chunks:
                entity = chunk["entity_name"]
                dates_list = chunk["dates"]
                dates_str = ', '.join(dates_list) if dates_list else 'нет дат'
                context_lines.append(f"Сущность: {entity}, даты: {dates_str}")
            context = "\n".join(context_lines)

    logger.info(f"Контекст сформирован, длина {len(context)} символов")

    # 5. Генерация ответа
    answer = await asyncio.to_thread(_generate_answer, query, context)
    return answer