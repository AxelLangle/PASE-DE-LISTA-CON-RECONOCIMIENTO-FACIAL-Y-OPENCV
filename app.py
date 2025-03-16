from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import os
import imutils
import threading
import base64

app = Flask(__name__)
socketio = SocketIO(app)

dataPath = 'C:/Users/chang/OneDrive - Universidad Politecnica de Tecamac/Escritorio/Reconocimiento Facial/Data'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/formulario')
def formulario():
    return render_template('formulario.html')

@app.route('/confirm', methods=['POST'])
def confirm():
    personName = f"{request.form['nombre']} {request.form['apellido_paterno']} {request.form['apellido_materno']}"
    return render_template('confirm.html', personName=personName)

@app.route('/capture', methods=['POST'])
def capture():
    personName = request.form['personName']
    return render_template('capture.html', personName=personName)

def capture_faces(personName):
    personPath = os.path.join(dataPath, personName)
    if not os.path.exists(personPath):
        os.makedirs(personPath)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 0
  
    while True:
        ret, frame = cap.read()
        if ret == False: 
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
            count = count + 1

        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('frame', {'frame': frame_encoded})

        k = cv2.waitKey(1)
        if k == 27 or count >= 500:
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/start_capture', methods=['POST'])
def start_capture():
    personName = request.form['personName']
    threading.Thread(target=capture_faces, args=(personName,)).start()
    return jsonify({'status': 'started'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
