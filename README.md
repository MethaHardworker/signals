# Постановка задачи:
Имеется набор фотографий аудиогидов и картонных коробок. Задача - определить можно ли поместить аудиогид в коробку использую только параллелльный перенос (без вращений).
Требования к датасету:
- коробка и аудиогид полностью помещаются на изображении
- коробка и аудиогид находятся на одной поверхности
- коробка и аудиогид не перекрывают друг друга
- других объектов, кроме коробки и аудиогида, на изображении быть не должно
- фон для фотографий белый (использовалась белая простыня)
- фотографии сделаны почти перпендикулярно к поверхности коробки, то есть так, чтобы боковые поверхности были не видны (возможны исключения и небольшая часть все же видна)
- качество фото не менее 8 МП
- на аудиогиде и коробки возможны засветы и блики от освещения
- оси коробки сонаправлены с осями фотографии (можно предположить, что исходные данные были предобработаны и фото повернуты в нужную сторону для выполнения этого условия)
- расстояние от камеры до коробки и до аудиогида примерно одинаковое, то есть пространственные искажения отсутствуют

# План:
## Аудиогид
- Преобразуем изображение в черно-белый формат. Т.к. аудиогид темнее остальных объектов, то будем его искать следующим образом. Оценим результаты работы всех фильтров яркости (из библиотеки sklearn) и выберем наиболее подходящий. Поскольку все фильтры могут давать неправильные результаты при наличии теней по краям изображения или выделять тень от коробки, как отдельную компоненту связности, то стоит использовать эмперически подобранный фильтр черного. 
Задается некий уровень значимости, меньше которого пиксель считается достаточно темным. Применяем этот фильтр к черно-белому изображению. 
После чего для улучшения результата используем морфологические фильтры (opening, closing)
- Аудиогид является самым крупным объектом из полученных, поэтому ищем наибольшую компоненту связности и находим ее границы, таким образом получим крайние границы прямоугольника, в который вписывается аудиогид.

## Коробка
- Также будем использовать черно-белое изображение. С помощью метода Canny находим все объекты на экране. Коробка больше аудиогида, поэтому компонента с наибольшей площадью будет коробкой. Снова находим крайние точки и размеры описанного прямоугольника.

## Принятие решения
- Сравниваем размеры 2 прямоугольников. Если размеры прямоугольника, описанного вокруг аудиогида больше, то решаем, что не поместится

# Пример работы алгоритма
![image](https://user-images.githubusercontent.com/55626617/112727746-0384b700-8f35-11eb-8915-8a0156782e49.png)

# Изменения на итерации 2
- Расширился датасет, в него добавились фотографии с коробкой, которая повернута относительно осей изображения

- Добавилась функция приведения новых изображений к "стандарному" виду

- Рефакторинг кода запуска проверки тестов

- Общий рефакторинг кода

# Результаты
Right negatives: 20 / 26

Right positives: 21 / 23
