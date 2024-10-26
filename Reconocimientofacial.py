import cv2
import os
import numpy as np
import mediapipe as mp

# Configuración de rutas y modelo
dataPath = 'C:/Users/Pablo/Desktop/vision/Data'  # Cambia a la ruta donde tienes almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Cargar el modelo entrenado para el reconocimiento facial
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')  # Cargar el modelo previamente entrenado

# Inicializar mediapipe para el reconocimiento de esqueleto, manos y cara
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Clasificador para detección de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Captura de video desde la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a RGB y escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    auxFrame = gray.copy()

    # Detectar rostros en la imagen
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    # Detectar poses en la imagen (esqueleto)
    pose_result = pose.process(rgb_frame)
    hand_result = hands.process(rgb_frame)
    face_result = face_mesh.process(rgb_frame)

    # Dibujar el esqueleto y etiquetar partes clave si se detecta
    if pose_result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2))

        # Etiquetas de puntos específicos del cuerpo
        landmarks = pose_result.pose_landmarks.landmark

        # Coordenadas y etiquetas para cada parte del cuerpo
        body_parts = {
            'Cabeza': landmarks[0],
            'Hombro izquierdo': landmarks[11],
            'Hombro derecho': landmarks[12],
            'Mano izquierda': landmarks[15],
            'Mano derecha': landmarks[16],
            'Pie izquierdo': landmarks[27],
            'Pie derecho': landmarks[28]
        }

        # Dibujar y etiquetar las partes del cuerpo
        for part, landmark in body_parts.items():
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Ajuste específico para que "Cabeza" aparezca más arriba de la frente
            if part == 'Cabeza':
                y -= 50  # Ajuste hacia arriba en el eje Y

            cv2.putText(frame, part, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    # Dibujar el esqueleto completo de la mano si se detectan
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            # Dibujar la estructura de la mano completa
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2))

    # Dibujar y etiquetar puntos faciales específicos para ojos, nariz y boca
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            # Índices de los puntos de la cara
            puntos_cara = {
                'Ojo derecho': 33, 'Ojo izquierdo': 263,
                'Nariz': 1, 'Boca': 13
            }

            # Dibujar y etiquetar puntos clave en la cara
            for part, idx in puntos_cara.items():
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                cv2.putText(frame, part, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    # Procesar y reconocer rostros detectados
    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        # Determinar si el rostro es conocido o desconocido
        if result[1] < 60:  # Ajustar el umbral según pruebas
            # Verifica que el índice esté dentro del rango
            if result[0] < len(imagePaths):
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde para rostro reconocido
            else:
                cv2.putText(frame, 'Error en ID', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rojo para desconocido
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rojo para desconocido

    # Mostrar el frame con reconocimiento facial, esqueleto y manos completas
    cv2.imshow('Reconocimiento Facial, Esqueleto y Partes Faciales', frame)

    # Salir del bucle con la tecla 'Esc'
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
