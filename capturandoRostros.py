import cv2
import os
import imutils

# Nombre de la persona y ruta donde se almacenarán las imágenes
personName = 'harby'
dataPath = 'C:/Users/Pablo/Desktop/vision/Data'  # Cambia a la ruta donde tengas almacenado Data
personPath = os.path.join(dataPath, personName)

# Crear la carpeta si no existe
if not os.path.exists(personPath):
    print(f'Carpeta creada: {personPath}')
    os.makedirs(personPath)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargar clasificadores para detectar rostros y cuerpos
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
bodyClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

count = 0  # Contador para el número de imágenes capturadas

# Bucle para capturar imágenes en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    # Detectar rostros en la imagen
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    # Detectar cuerpos en la imagen
    bodies = bodyClassif.detectMultiScale(gray, 1.1, 3)

    # Guardar imágenes de rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(personPath, 'rostro_{}.jpg'.format(count)), rostro)
        count += 1

    # Guardar imágenes de cuerpos detectados
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cuerpo = auxFrame[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(personPath, 'cuerpo_{}.jpg'.format(count)), cuerpo)
        count += 1

    # Mostrar el frame
    cv2.imshow('frame', frame)

    # Salir del bucle si se presiona 'Esc' o si se capturan 300 imágenes
    k = cv2.waitKey(1)
    if k == 27 or count >= 600:
        break

cap.release()
cv2.destroyAllWindows()