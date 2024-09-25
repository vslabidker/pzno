import numpy as np

##Task 3

def find_best_row(seats):
    max_free_seats = 0  # Максимальное количество свободных мест подряд
    best_row = -1       # Номер ряда с максимальной последовательностью свободных мест

    # Проходим по каждому ряду
    for i, row in enumerate(seats):
        current_free_seats = 0  # Текущая длина последовательности свободных мест
        max_in_row = 0  # Максимальная последовательность свободных мест в данном ряду

        # Проходим по каждому месту в ряду
        for seat in row:
            if seat == 0:  # Место свободно
                current_free_seats += 1
                max_in_row = max(max_in_row, current_free_seats)
            else:  # Место занято
                current_free_seats = 0  # Сбрасываем счетчик

        # Если нашли ряд с большем количеством свободных мест подряд
        if max_in_row > max_free_seats:
            max_free_seats = max_in_row
            best_row = i + 1  # Ряды нумеруются с 1, а не с 0

    return best_row




seats = np.random.choice([0, 1], size=(15, 30))
print(seats)

best_row = find_best_row(seats)
print(f'Лучший ряд для посадки: {best_row}')

##Task 4

def calculate_word_percentages(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Разделяем текст на слова, убирая знаки пунктуации
    words = [word.strip(' .,!?;:()[]\'"') for word in text.split()]
    total_words = len(words)

    if total_words == 0:
        return 0, 0  # Если слов нет, возвращаем 0%

    # Подсчет коротких и длинных слов
    short_words = [word for word in words if len(word) < 4]
    long_words = [word for word in words if len(word) > 10]

    # Процент коротких и длинных слов
    short_words_percent = len(short_words) / total_words * 100
    long_words_percent = len(long_words) / total_words * 100

    return short_words_percent, long_words_percent


# Пример использования
file_path = 'input.txt'  # Путь к файлу
short_percent, long_percent = calculate_word_percentages(file_path)

print(f"Процент коротких слов: {short_percent:.0f}%")
print(f"Процент длинных слов: {long_percent:.0f}%")
