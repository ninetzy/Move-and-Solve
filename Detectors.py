import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class JumpCounter:
    def __init__(self):
        self.jump_count = 0  # Прыжки
        self.is_in_air = False  # Сейчас в воздухе?
        self.base_height = None  # Высота, когда стоит на земле

        # Пороги для определения прыжка
        self.JUMP_HEIGHT_THRESHOLD = 0.03  # Минимальная высота прыжка
        self.GROUND_HEIGHT_THRESHOLD = 0.01  # Порог для определения "на земле"

    def calculate_hip_height(self, landmarks):
        left_hip = landmarks.landmark[23]  # Левое бедро
        right_hip = landmarks.landmark[24]  # Правое бедро
        hip_y = (left_hip.y + right_hip.y) / 2 # Среднее положение бедер по y-координате
        return hip_y

    def detect_jump(self, current_height):
        if self.base_height is None:
            self.base_height = current_height
            return False
        height_diff = self.base_height - current_height
        if abs(height_diff) < self.GROUND_HEIGHT_THRESHOLD:
            self.base_height = current_height
            self.is_in_air = False
        if not self.is_in_air and height_diff > self.JUMP_HEIGHT_THRESHOLD:
            self.is_in_air = True
            return True
        if self.is_in_air and abs(height_diff) < self.GROUND_HEIGHT_THRESHOLD:
            self.is_in_air = False
        return False
    def update(self, landmarks, frame_height):
        current_height = self.calculate_hip_height(landmarks)
        if self.detect_jump(current_height):
            self.jump_count += 1
        return self.jump_count

