FROM tensorflow/tensorflow:2.13.0

RUN apt-get update && apt-get install -y libgl1-mesa-dev python3-pip
RUN pip install -U tensorflow_hub flask flask_restful numpy opencv-python Pillow flask-cors

COPY . .

CMD ["python", "main.py"]