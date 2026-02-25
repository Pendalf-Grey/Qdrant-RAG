import os
import re


def split_entities(input_file: str, output_dir: str):
    """Разбивает файл с сущностями на отдельные файлы."""
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Регулярное выражение для поиска блоков сущностей
    # Ищем имя в начале строки, затем строку "Даты:", затем даты до следующего имени или конца файла
    pattern = re.compile(r'^([a-z]+) \([^)]+\)\nДаты:\n((?:\s+\d{2}\.\d{2}\n?)+)', re.MULTILINE)

    for match in pattern.finditer(content):
        entity_name = match.group(1)  # латинское имя
        dates_block = match.group(2).strip()
        # Извлекаем все даты (они уже в нужном формате)
        dates = dates_block.split()

        # Формируем содержимое нового файла: первая строка — имя, затем каждая дата на новой строке
        output_content = f"{entity_name}\n" + "\n".join(dates)

        output_file = os.path.join(output_dir, f"{entity_name}.txt")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(output_content)
        print(f"Создан файл: {output_file}")


if __name__ == "__main__":
    input_file = "date_in_format_dd_mm.txt"  # путь к вашему большому файлу
    output_dir = "entities"
    split_entities(input_file, output_dir)