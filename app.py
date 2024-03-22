import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import Flask, request, render_template , jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

model_path = 'quantized_model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print('Model loaded. Check http://127.0.0.1:5000/')

# Define labels in the same order as the model output
labels = {0:'Apple___Apple_scab', 1:'Apple___Black_rot',2:'Apple___Cedar_apple_rust',3:'Apple___healthy',4:'Blueberry___healthy',5:'Cherry_(including_sour)___Powdery_mildew',6:'Cherry_(including_sour)___healthy',7:'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',8:'Corn_(maize)___Common_rust_',9:'Corn_(maize)___Northern_Leaf_Blight',10:'Corn_(maize)___healthy',11:'Grape___Black_rot',12:'Grape___Esca_(Black_Measles)',13:'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',14:'Grape___healthy',15:'Orange___Haunglongbing_(Citrus_greening)',16:'Peach___Bacterial_spot',17:'Peach___healthy',18:'Pepper,_bell___Bacterial_spot',19:'Pepper,_bell___healthy',20:'Potato___Early_blight',21:'Potato___Late_blight',22:'Potato___healthy',23:'Raspberry___healthy',24:'Soybean___healthy',25:'Squash___Powdery_mildew',26:'Strawberry___Leaf_scorch',27:'Strawberry___healthy',28:'Tomato___Bacterial_spot',29:'Tomato___Early_blight',30:'Tomato___Late_blight',31:'Tomato___Leaf_Mold',32:'Tomato___Septoria_leaf_spot',33:'Tomato___Spider_mites Two-spotted_spider_mite',34:'Tomato___Target_Spot',35:'Tomato___Tomato_Yellow_Leaf_Curl_Virus',36:'Tomato___Tomato_mosaic_virus',37:'Tomato___healthy'}

def getResult(image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = load_img(image_path, target_size=(input_details[0]['shape'][1], input_details[0]['shape'][2]))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    confidence_score = np.max(predictions)
    print("Confidence score:", confidence_score)
    return predictions,confidence_score

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predictApi', methods=["POST"])
def api():
    try:
        if 'file' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('file')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)

        predictions, confidence_score = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        if(confidence_score<0.9):
            predicted_label="Cannot Detect Disease"
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'Error': str(e)})

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        predictions, confidence_score = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        if(confidence_score<0.9):
            predicted_label="Cannot Detect Disease"

        return predicted_label

    return None

if __name__ == '__main__':
    app.run(debug=True)
