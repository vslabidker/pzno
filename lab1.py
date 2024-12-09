import numpy as np
##Task 1
def calculate_e(epsilon):
    x = 1  
    prev_l = 0  
    
    while True:
        current_l = (1 + x) ** (1 / x)
        if abs(current_l - prev_l) < epsilon:
            return current_l
        prev_l = current_l
        x /= 2  
        
epsilon = 1e-6 
e_approx = calculate_e(epsilon)
print(f"Число e с точностью {epsilon}: {e_approx}")

##Task 2
def process_phone_number(phone_number):
    digits = list(str(phone_number))
    for i in range(len(digits) - 1, -1, -1):
        if int(digits[i]) % 2 != 0:  
            result_number = int("".join(digits[:i + 1]))
            return result_number
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True
    
phone_number = 123456789 
result = process_phone_number(phone_number)
print(f"Результирующее число: {result}")
print(f"Является простым:{is_prime(result)}")


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

    # Разделяем текст на слова, убирая пробелы, знаки пунктуации, если такие присутствуют
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

file_path = 'input.txt'  # Путь к файлу
short_percent, long_percent = calculate_word_percentages(file_path)

print(f"Процент коротких слов: {short_percent:.0f}%")
print(f"Процент длинных слов: {long_percent:.0f}%")
