from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("plant_model.h5")

classes = [
'Pepper__bell___Bacterial_spot',
'Pepper__bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Tomato_Bacterial_spot',
'Tomato_Early_blight',
'Tomato_Late_blight',
'Tomato_Leaf_Mold',
'Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites',
'Tomato_Target_Spot',
'Tomato_YellowLeaf_Curl_Virus',
'Tomato_mosaic_virus',
'Tomato_healthy'
]

solutions = {
'Tomato_Early_blight':"Use Mancozeb fungicide spray",
'Tomato_Late_blight':"Use Copper fungicide",
'Potato_Early_blight':"Remove infected leaves and spray fungicide",
'Potato_Late_blight':"Use metalaxyl fungicide",
'Tomato_healthy':"Plant is healthy"
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)/255

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    disease = classes[class_index]
    solution = solutions.get(disease,"Consult agricultural expert")

    result = "Disease: "+disease+" | Solution: "+solution

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)