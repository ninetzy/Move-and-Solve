import cv2
import mediapipe as mp
from Detectors import JumpCounter, SquatCounter

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
jump_counter = JumpCounter()
squat_counter = SquatCounter()
last_jump_count = 0
last_squat_count = 0

# Функция для подсчета движений
def movements_counter():
    # Используем глобальные переменные last_jump_count и last_squat_count
    global last_jump_count, last_squat_count
    # Получаем изображение с камеры
    _, frame = cap.read()

    # Переводим в RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Обнаружение ключевых точек человека
    results = pose.process(frame_rgb)

    # Если найдены ключевые точки (в кадре есть человек)
    if results.pose_landmarks:
        # С помощью класса JumpCounter получаем кол-во сделанных прыжков
        current_jump_count = jump_counter.update(results.pose_landmarks)

        # С помощью класса SquatCounter получаем кол-во сделанных приседаний
        current_squat_count = squat_counter.update(results.pose_landmarks)

        # Если кол-во прыжков изменилось с прошлого кадра
        if current_jump_count != last_jump_count:
            # Записываем текущее кол-во прыжков в last_jump_count и выводим новое кол-во прыжков
            last_jump_count = current_jump_count
            print(f'Прыжки - {current_jump_count}')

        # Если кол-во приседаний изменилось с прошлого кадра
        if current_squat_count != last_squat_count:
            # Записываем текущее кол-во приседаний в last_squat_count и выводим новое кол-во приседаний
            last_squat_count = current_squat_count
            print(f'Приседания - {current_squat_count}')

        # Рисуем ключевые точки на изображении камеры (для удобства)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    # Если человек не в кадре
    else:
        # Сбрасываем стартовые переменные до появления человека в кадре
        jump_counter.start_height = None
        jump_counter.is_in_air = False
        squat_counter.start_height = None
        squat_counter.is_down = False

    # Демонстрация изображения с камеры
    cv2.imshow('Camera', frame)

# Основной цикл
while True:
    # Запускаем функцию подсчета движений
    movements_counter()
    cv2.waitKey(1)

# Выключаем камеру и закрываем окна
cap.release()
cv2.destroyAllWindows()