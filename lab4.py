import pandas as pd
import matplotlib.pyplot as plt

## Task 1
vac = pd.read_excel("vaccination_process_2021_regions.xlsx", sheet_name='daily_vaccination process quant')
vac_od = vac[vac["Назва території"]=="Одеська область"]
vac_od["Дата (період) данних"] = pd.to_datetime(vac_od["Дата (період) данних"], dayfirst=True)
vac_od_pf = vac_od.groupby("Дата (період) данних").sum()
vac_od_pf = vac_od_pf.resample('M').sum()

plt.figure(figsize=(10, 6))
plt.bar(vac_od_pf.index.strftime('%m'), vac_od_pf['Pfizer-BioNTech, осіб'], color='skyblue')

plt.title("Pfizer-BioNTech в Одеській області за 2021 рік")
plt.xlabel("Місяць")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
# print(vac_od_pf.to_string())

## Task 2

flat = pd.read_excel("filter1.xlsx", sheet_name="filter1")

flat['Surplus'] = flat['Meters'] - (flat['Quantity_of_people'] * 21 + 10)
flat['Surplus'] = flat['Surplus'].apply(lambda x: max(0, x))  # Если излишек отрицательный, ставим 0

cond1 = flat[flat['Privileges'] == 'yes']

cond2 = flat[(flat['Privileges'] == 'yes') & (flat['Surplus'] == 0)]

cond3 = flat[flat['Meters'] > 100]

cond4 = flat[(flat['Year_of_registration'] >= 1970) & (flat['Year_of_registration'] <= 1990)]

cond5 = flat[(flat['Privileges'] == 'yes') & (flat['Year_of_registration'] > 1980)]

cond6 = flat[
    ((flat['Privileges'] == 'yes') & (flat['Surplus'] > 0)) |
    ((flat['Meters'] < 100) & (flat['Year_of_registration'] < 1977))
]

cond7 = flat[
    ((flat['Surplus'] > 50) & (flat['Privileges'] == 'no')) |
    ((flat['Year_of_registration'] >= 1970) & (flat['Year_of_registration'] <= 1990) & (flat['Meters'] > 100))
]

print("1. Квартиросъёмщики, имеющие льготы")
print(cond1.to_string())

print("\n2. Квартиросъёмщики, имеющие льготы и не имеющие излишков площади")
print(cond2.to_string())

print("\n3. Квартиросъёмщики, чей общий метраж превышает 100 м²")
print(cond3.to_string())

print("\n4. Квартиросъёмщики, прописавшиеся в период с 1970 по 1990 год")
print(cond4.to_string())

print("\n5. Квартиросъёмщики, имеющие льготы и прописавшиеся после 1980 года")
print(cond5.to_string())

print("\n6. Льготники с избытками площади или те, кто зарегистрирован до 1977 года и имеет метраж менее 100 м²")
print(cond6.to_string())

print("\n7. Лица с излишками более 50 м² и без льгот или зарегистрированные в 1970-1990 и имеющие метраж > 100 м²")
print(cond7.to_string())

## Task 3

# Загрузка данных из файлов
students1 = pd.read_csv('students1.txt', delimiter='\t')
students2 = pd.read_csv('students2.txt', delimiter='\t')
students3 = pd.read_csv('students3.txt', delimiter='\t')

# Преобразуем недостающие значения в NaN для корректной обработки
students1.replace(['*', ''], pd.NA, inplace=True)
students2.replace(['*', ''], pd.NA, inplace=True)
students3.replace(['*', ''], pd.NA, inplace=True)

# Объединение таблиц
students = pd.merge(students1, students2, on="Name", how="outer")
students = pd.merge(students, students3, on="Name", how="outer")

# Выбор оценок по предметам для задачи 1 и вычисление коэффициентов корреляции
math_analysis = students[['Математичний аналіз1', 'Математичний аналіз2', 'Математичний аналіз3']].astype(float)
corr_math_analysis = math_analysis.corr()

# Выбор оценок по предметам для задачи 2 и вычисление коэффициентов корреляции
history_ml = students[['Історія України', 'Машинне навчання']].astype(float)
corr_history_ml = history_ml.corr()

# Вывод результатов
print("Коэффициенты корреляции между оценками по математическому анализу (попарно):")
print(corr_math_analysis)
print("\nКоэффициент корреляции между Историей Украины и Машинным обучением:")
print(corr_history_ml)


## Task 4

flights_df = pd.read_csv('2008_rand.csv')
airports_df = pd.read_csv('airports.csv')

average_distance = flights_df['Distance'].mean()
min_distance = flights_df['Distance'].min()
max_distance = flights_df['Distance'].max()

print(f'Средняя дистанция: {average_distance} \nМинимальная дистанция: {min_distance} \nМаксимальная дистанция: {max_distance}')

min_distance_flights = flights_df[flights_df['Distance'] == min_distance]

unique_flights_with_min_distance = min_distance_flights['FlightNum'].unique()
other_days_min_distance_flights = flights_df[flights_df['FlightNum'].isin(unique_flights_with_min_distance)]

print(min_distance_flights[['FlightNum', 'Year', 'Month', 'DayofMonth', 'Distance']], other_days_min_distance_flights[['FlightNum', 'Year', 'Month', 'DayofMonth', 'Distance']].drop_duplicates())

flights_by_month = flights_df['Month'].value_counts().idxmax()
print(f'Месяц с наибольшим количеством полетов: {flights_by_month}')

december_flights = flights_df[flights_df['Month'] == 12]
top_5_destinations = december_flights['Dest'].value_counts().head(5)

top_5_destinations_df = airports_df[airports_df['iata'].isin(top_5_destinations.index)]
top_5_destinations_with_cities = top_5_destinations_df[['iata', 'city']].set_index('iata').reindex(top_5_destinations.index)

max_dep_delay_flight = flights_df.loc[flights_df['DepDelay'].idxmax()]
max_dep_delay_info = {
    'airport': max_dep_delay_flight['Origin'],
    'delay_minutes': max_dep_delay_flight['DepDelay'],
    'year': max_dep_delay_flight['Year'],
    'month': max_dep_delay_flight['Month'],
    'day': max_dep_delay_flight['DayofMonth']
}

print(top_5_destinations_with_cities.to_string(), max_dep_delay_info)
