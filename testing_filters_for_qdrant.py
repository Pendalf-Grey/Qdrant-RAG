# from qdrant_client import QdrantClient, models
#
# client = QdrantClient("localhost", port=6333)
# my_strings = ["неммамиах", "иейазель", "orange"]
# all_points = []
# offset = None
#
# while True:
#     # Получаем страницу результатов
#     result = client.scroll(
#         collection_name="legal_docs",
#         scroll_filter=models.Filter(
#             should=[
#                 models.FieldCondition(
#                     key="entity_name",
#                     match=models.MatchValue(value=s)
#                 ) for s in my_strings
#             ]
#         ),
#         limit=100,  # размер страницы
#         offset=offset,  # со следующей страницы передаем offset от предыдущего ответа
#         with_payload=True
#     )
#
#     points = result[0]  # список точек на текущей странице
#     all_points.extend(points)
#
#     # Получаем offset для следующей страницы
#     offset = result[1]  # если None — страниц больше нет
#     if offset is None:
#         break
#
# print(f"Всего найдено чанков: {len(all_points)}")
# print(f"Всего найдено чанков: {all_points}")



# count_result = client.count(
#     collection_name="legal_docs",
#     count_filter=models.Filter(
#         should=[
#             models.FieldCondition(
#                 key="entity_name",
#                 match=models.MatchValue(value=s)
#             ) for s in my_strings
#         ]
#     ),
#     exact=True  # точный подсчет (не приблизительный) [citation:6]
# )
#
# print(f"Количество чанков, содержащих искомые строки: {count_result.count}")


'''Создадим индексацию всех чанков, чтобы потом фильтровать по MatchText'''
from qdrant_client import QdrantClient, models

client = QdrantClient("localhost", port=6333)

# Создаем полнотекстовый индекс для поля chunk_text
client.create_field_index(
    collection_name="legal_docs",
    field_name="chunk_text",
    field_type=models.FieldType.TEXT,
    field_index_params=models.TextIndexParams(
        type="text",
        tokenizer=models.TokenizerType.WORD,  # разбивка на слова
        min_token_len=2,  # минимальная длина слова для индексации
        max_token_len=10,  # максимальная длина слова
        lowercase=True  # приводить к нижнему регистру
    )
)