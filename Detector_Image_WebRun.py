import os
import cv2
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (70, 70)
UPLOAD_FOLDER = r"C:\Users\sungw\Projects\EmptyShelf_Detection\static\Upload"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(img,(70,70)) 
    ex = np.array(new_img).reshape(-1,70,70,1)
    ex = ex / 255.0
 
    model = load_model(r'C:\Users\sungw\Projects\EmptyShelf_Detection\empty_v2_K-Fold.model')
    prediction = model.predict(ex)
    return prediction


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', image_file='Home.jpg')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            output = predict(file_path)

            if output < 0.5:
                pred_class = "Empty      - probalility : "
            else:
                pred_class = "Not Empty  - probalility : "
            
            output = pred_class + str(output)

    return render_template("home.html", label=output, image_file=filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)