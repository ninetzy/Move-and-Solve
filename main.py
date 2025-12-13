import cv2
import mediapipe as mp
from Detectors import JumpCounter, SquatCounter, BendCounter
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

MODEL_PATH = "pose_landmarker_full.task"

# Загружаем модель в буфер, так как обычная не раотает на windows
with open(MODEL_PATH, "rb") as f:
    model_buffer = f.read()

# Настраиваем детектор позы, чтобы обрабатывал только 3 человека
base_options = python.BaseOptions(model_asset_buffer=model_buffer)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=3,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Список с данными для каждого человека
people_data = []

# Функция для подсчета движений
def movements_counter():
    # Используем глобальную переменную people_data
    global people_data

    # Получаем изображение с камеры
    success, frame = cap.read()

    # Переводим в RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Создаем объект Image для MediaPipe Tasks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Находим все позы людей в кадре с помощью Tasks
    detect_result = detector.detect(mp_image)

    # Если найдены ключевые точки (в кадре есть люди)
    if detect_result.pose_landmarks:
        # Определяем количество людей в кадре
        num_people = len(detect_result.pose_landmarks)

        # Если количество людей изменилось с прошлого кадра
        if len(people_data) != num_people:
            # Создаем новые данные для каждого человека
            people_data = []
            for i in range(num_people):
                people_data.append({
                    # Счетчик прыжков для этого человека
                    'jump_counter': JumpCounter(),
                    # Счетчик приседаний для этого человека
                    'squat_counter': SquatCounter(),
                    # Счетчик наклонов для этого человека
                    'bend_counter': BendCounter(),
                    # Последнее зафиксированное количество прыжков
                    'last_jump_count': 0,
                    # Последнее зафиксированное количество приседаний
                    'last_squat_count': 0,
                    # Последнее зафиксированное количество наклонов
                    'last_bend_count': 0
                })

        # Циклом проходимся по каждому человеку
        for person_id in range(num_people):
            # Получаем ключевые точки для человека, которого обрабатываем
            landmarks = detect_result.pose_landmarks[person_id]
            person_data = people_data[person_id]

            # С помощью класса JumpCounter получаем кол-во сделанных прыжков
            current_jump_count = person_data['jump_counter'].update(landmarks)

            # С помощью класса SquatCounter получаем кол-во сделанных приседаний
            current_squat_count = person_data['squat_counter'].update(landmarks)

            # С помощью класса BendCounter получаем кол-во сделанных наклонов
            current_bend_count = person_data['bend_counter'].update(landmarks)

            # Если кол-во прыжков изменилось с прошлого кадра
            if current_jump_count != person_data['last_jump_count']:
                # Записываем текущее кол-во прыжков в last_jump_count для человека, которого
                # обрабатываем, и выводим новое кол-во прыжков
                person_data['last_jump_count'] = current_jump_count
                print(f'Человек #{person_id + 1} - Прыжки: {current_jump_count}')

            # Если кол-во приседаний изменилось с прошлого кадра
            if current_squat_count != person_data['last_squat_count']:
                # Записываем текущее кол-во приседаний в last_squat_count для человека, которого
                # обрабатываем, и выводим новое кол-во приседаний
                person_data['last_squat_count'] = current_squat_count
                print(f'Человек #{person_id + 1} - Приседания: {current_squat_count}')

            # Если кол-во наклонов изменилось с прошлого кадра
            if current_bend_count != person_data['last_bend_count']:
                # Записываем текущее кол-во наклонов в last_bend_count для человека, которого
                # обрабатываем, и выводим новое кол-во наклонов
                person_data['last_bend_count'] = current_bend_count
                print(f'Человек #{person_id + 1} - Наклоны: {current_bend_count}')

            # Создаем NormalizedLandmarkList для отрисовки, потому что Tasks возвращает
            # обычный список питон, а функция draw_landmarks требует специальный формат.
            # Для этого используем landmark_pb2
            from mediapipe.framework.formats import landmark_pb2
            landmark_list_proto = landmark_pb2.NormalizedLandmarkList()
            landmark_list_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility
                ) for landmark in landmarks
            ])

            # Рисуем ключевые точки на изображении камеры (для удобства)
            mp_drawing.draw_landmarks(
                frame,
                landmark_list_proto,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
    # Если человек не в кадре
    else:
        # Сбрасываем данные для всех людей
        if people_data:
            people_data = []

    # Демонстрация изображения с камеры
    cv2.imshow('Camera - Несколько людей', frame)

# Основной цикл
while True:
    # Запускаем функцию подсчета движений
    movements_counter()
    cv2.waitKey(1)

# Выключаем камеру и закрываем окна
cap.release()
cv2.destroyAllWindows()

# Закрываем детектор Tasks
detector.close()