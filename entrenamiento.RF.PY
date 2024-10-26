#entrenamiento.RF
import cv2
import os
import numpy as np

# Definir la ruta donde se almacenan las imágenes
dataPath = 'C:/Users/Pablo/Desktop/vision/Data'  # Cambia a la ruta donde tienes almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

# Variables para almacenar los datos de las imágenes y sus etiquetas
labels = []
facesData = []
label = 0

# Leer las imágenes de cada persona/categoría (rostros o cuerpos)
for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo las imágenes de: ', nameDir)

    for fileName in os.listdir(personPath):
        print('Rostros/Cuerpos: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
    label += 1

# Entrenar el modelo LBPH con las imágenes recogidas
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Entrenando el modelo...")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado en un archivo
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado correctamente.")