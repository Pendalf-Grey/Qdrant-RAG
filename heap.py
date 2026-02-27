import os
import re
import hashlib
from typing import List
import requests
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf


# ====================== ИЗВЛЕЧЕНИЕ ДАТ (REGEX) ======================
def extract_dates_regex(text: str) -> List[str]:
    pattern = r'\b(\d{2})\.(\d{2})\b'
    matches = re.findall(pattern, text)
    dates = [f"{d}.{m}" for d, m in matches]
    return list(set(dates))


# ====================== ПАРСИНГ ФАЙЛА СУЩНОСТИ ======================
def parse_entity_file(file_path: str):
    """
    Читает файл, где первая строка — имя сущности,
    остальные строки — даты (каждая на новой строке).
    Возвращает (entity_name, list_of_dates, full_text)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None, [], ""
    entity_name = lines[0]
    dates = lines[1:]  # все остальные строки — даты
    # Проверим, что все они в формате ДД.ММ (для надёжности)
    valid_dates = [d for d in dates if re.match(r'^\d{2}\.\d{2}$', d)]
    # Полный текст можно собрать из дат, разделённых пробелами
    full_text = " ".join(valid_dates)
    return entity_name, valid_dates, full_text


# ====================== ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ ФАЙЛА (общая) ======================
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        text = ""
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")


# ====================== ЭМБЕДДИНГ ЧЕРЕЗ OLLAMA ======================
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        embeddings.append(response.json()["embedding"])
    return embeddings


# ====================== ГЕНЕРАЦИЯ ID ДОКУМЕНТА ======================
def get_document_id(file_path: str) -> str:
    abs_path = os.path.abspath(file_path)
    return hashlib.sha256(abs_path.encode('utf-8')).hexdigest()


# ====================== РАБОТА С QDRANT ======================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_docs"
VECTOR_SIZE = 768


def create_collection_if_not_exists(client: QdrantClient):
    collections = client.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10000,
                memmap_threshold=20000
            )
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="dates",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="entity_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print(f"Коллекция '{COLLECTION_NAME}' создана.")
    else:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует.")


# ====================== ОСНОВНАЯ ФУНКЦИЯ ИНДЕКСАЦИИ ======================
def index_document(
        client: QdrantClient,
        file_path: str,
        point_counter: int,
        batch_size: int = 10
) -> int:
    print(f"Обработка: {file_path}")

    # Для маленьких файлов мы не разбиваем на чанки (или можно разбить, если дат очень много)
    # Но в данном случае файл содержит одну сущность и несколько дат – можно не чанковать.
    entity_name, dates, full_text = parse_entity_file(file_path)
    if not entity_name:
        print(f"  Файл не содержит корректных данных, пропускаем.")
        return point_counter

    # Если дат нет, пропускаем
    if not dates:
        print(f"  Нет дат в файле, пропускаем.")
        return point_counter

    # Для эмбеддинга используем полный текст (строку с датами)
    # Можно также добавить имя сущности в текст, чтобы улучшить поиск
    text_for_embedding = f"{entity_name} {full_text}"

    embeddings = get_embeddings([text_for_embedding])
    embedding = embeddings[0]

    # Формируем точку
    point_id = point_counter
    point_counter += 1

    doc_id = get_document_id(file_path)

    payload = {
        "document_id": doc_id,
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "entity_name": entity_name,
        "dates": dates,  # теперь это просто список дат
        "text": full_text  # можно сохранить текст для вывода
    }

    points = [models.PointStruct(
        id=point_id,
        vector=embedding,
        payload=payload
    )]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"    - Загружена 1 точка для {entity_name}")

    print(f"  ✅ Готово, следующий point_id = {point_counter}\n")
    return point_counter


# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    create_collection_if_not_exists(client)

    # Очищаем коллекцию перед тестом (опционально)
    print("Очищаем коллекцию...")
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.Filter(
            must=[]  # удаляет все точки
        )
    )
    print("Коллекция очищена.")

    point_counter = 0
    # Папка с файлами сущностей
    entities_folder = "data/entities_5_dates"
    file_list = []
    for root, dirs, files in os.walk(entities_folder):
        for file in files:
            if file.endswith('.txt'):
                file_list.append(os.path.join(root, file))

    for file_path in file_list:
        try:
            point_counter = index_document(client, file_path, point_counter)
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {e}")

    print("Индексация завершена.")