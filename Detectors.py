import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class JumpCounter:
    def __init__(self):
        # Прыжки
        self.jump_count = 0

        # Сейчас в воздухе?
        self.is_in_air = False

        # Высота, когда стоит на земле
        self.start_height = None

        # Пороги для определения прыжка
        # Минимальная высота прыжка
        self.JUMP_HEIGHT_THRESHOLD = 0.04

        # Порог для определения, что человек на земле
        self.GROUND_HEIGHT_THRESHOLD = 0.01

    def calculate_hip_height(self, landmarks):
        # Левое бедро
        left_hip = landmarks.landmark[23]

        # Правое бедро
        right_hip = landmarks.landmark[24]

        # Среднее положение бедер по y-координате
        hip_y = (left_hip.y + right_hip.y) / 2
        return hip_y

    def detect_jump(self, current_height):
        # Если это первый кадр, иницилизируем начальную высоту
        if self.start_height is None:
            self.start_height = current_height
            return False

        # Находим разницу высот
        height_diff = self.start_height - current_height

        # Если человек стоит на земле, обновляем начанльную высоту
        if abs(height_diff) < self.GROUND_HEIGHT_THRESHOLD:
            self.start_height = current_height
            self.is_in_air = False

        # Если человек был не в воздухе (то есть этот прыжок еще не был посчитан)
        # и сейчас он находится на высоте прыжка,
        # то возвращаем, что человек находится в прыжке
        if not self.is_in_air and height_diff > self.JUMP_HEIGHT_THRESHOLD:
            self.is_in_air = True
            return True

        # Если человек находился в воздухе и сейчас он находится на земле,
        # то записываем, что он находится не в прыжке
        if self.is_in_air and abs(height_diff) < self.GROUND_HEIGHT_THRESHOLD:
            self.is_in_air = False

        # Возвращаем, что человек не в прыжке, если условие того,
        # что он в прыжке, не выполнилось
        return False

    def update(self, landmarks):
        # Находим высоту, на которой сейчас человек
        current_height = self.calculate_hip_height(landmarks)

        # Если человек новом (не посчитанном) прыжке,
        # то добавляем его к итоговому кол-ву
        if self.detect_jump(current_height):
            self.jump_count += 1
        return self.jump_count

