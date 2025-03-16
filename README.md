# PASE-DE-LISTA-CON-RECONOCIMIENTO-FACIAL-Y-OPENCV
PASE-DE-LISTA-CON-RECONOCIMIENTO-FACIAL
Sistema creado como parte del proyecto de la materia Programación Cliente - Servidor. Tiene como objetivo el automatizar el pase de lista en las escuelas, haciendo asi mas agil el inicio de las clases de cada profesor.

Para la ejecucion de este programas se deben de seguir una serie de pasos antes:

1. INSTALACION DE LIBRERIAS EN "SIMBOLO DE SISTEMA" (O CMD) UTILIZANDO PIP
  - pip install Python
  - pip install cmake
  - pip install opencv-contrib-python
  - pip install face-recognition
  - pip install flask
  - pip install flask-socketio
  - pip install imutils
  - pip install dlib
    (PARA ESTE ANTERIOR SE DEBE DE TENER VISUAL STUDIO 2022 Y CUANDO SE INSTALE HABILITAR EL
    WINDOWS 10 JDK Y EL ENTORNO DE DESARROLLO DE C++)

2. TENER UN COMPILADOR DE CODIGO
   En este caso yo utilice VISUAL STUDIO CODE
   Creas una carpeta (con cualquier nombre), luego con el comando ctrl+K+O abres la carpeta
   Con el comando ctrl+J abres una terminal y seleccionas una terminal de git-bash En la
   terminal sigue los siguientes comando:
      - python -m venv .virtual
      - pip install flask
  Luego creas una carpeta llamada "templates" en la cual se guardaran todos los archivos .html   El archivo "app.py" se guarda en la caperta principal (no en la carpeta de templates ni en
  ninguna otra, sino en la carpeta del proyecto).
  CREACION DE CARPETA PARA GURADAR ARCHIVOS DEL RECONOCIMIENTO FACIAL En la ruta de tu
  preferencia crea una carpeta llamda "Reconocimiento Facial" y dentro de esta carpeta crea la
  carpeta "Data". La direccion de esta carpeta es la que pondras en la variable llamada
  "dataPath" la cual se utiliza en el codigo "app.py"
