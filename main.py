from app import app
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from scipy import spatial


from xgboost import XGBClassifier

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

mypath = 'static/output/'
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        query_filename = filename
        print('upload_image filename: ' + filename)
        # flash('Image successfully analysed and results are displayed below')
        with open('Mappings_folder.pickle', 'rb') as handle:
            mappings = pickle.load(handle)
        full_x = np.load('Paris_X_train.npy')
        print(full_x.shape)

        labels = np.load('Paris_label.npy')
        temp_count = 0
        count = {0: 0}
        j = 1
        for i in range(len(labels) - 1):
            if labels[i + 1] == labels[i]:
                temp_count += 1
            else:
                count[j] = temp_count + 1
                temp_count = temp_count + 1
                j += 1
        count[15] = full_x.shape[0] - 1
        print(count)
        paris_combined_lda = np.load('Paris_combined_LDA.npy')
        # paris_combined_pca = np.load('Paris_combined_5500.npy')
        paris_predict = np.load('Paris_combined_LDA.npy')
        paris_match = np.load('Paris_combined_2653.npy')
        print(paris_predict.shape)
        print(paris_match.shape)
        position = list(mappings.keys())[list(mappings.values()).index(str(filename))]
        print(position)

        index = [position]
        test_labels = []
        query_image = []
        query_image_match = []
        show = full_x[index[0]]

        for i in range(len(index)):
            query_image.append(paris_predict[index[i]])
            query_image_match.append(paris_match[index[i]])
        query_image = np.array(query_image)
        query_image_match = np.array(query_image_match)
        print(query_image.shape)
        print(query_image_match.shape)

        for i in range(len(index)):
            paris_match = np.delete(paris_match, index[i], axis=0)
            paris_predict = np.delete(paris_predict, index[i], axis=0)
            full_x = np.delete(full_x, index[i], axis=0)
        print(paris_predict.shape)

        for i in range(len(index)):
            test_labels.append(labels[index[i]])

        for i in range(len(index)):
            labels = np.delete(labels, index[i], axis=0)
        print(labels.shape)

        model = XGBClassifier()
        model.fit(paris_predict, labels)
        y_pred = model.predict(query_image)

        print(y_pred)
        acc = accuracy_score(y_pred, test_labels)
        print(acc)

        dist = {}
        for i in range(count[y_pred[0]], count[y_pred[0] + 1]):
            dist1 = spatial.distance.cosine(paris_match[i], query_image_match[0])
            dist[i] = dist1

        dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
        print(len(dist))
        print(dist)
        recipe=[]
        onlyfiles=[]
        recipe_information = pd.read_csv("dataset/Recipe_Mapping.csv")
        for i in range(1, 6):
            val = list(dist.keys())[i]
            if val > position:
                val1 = val + 1
            else:
                val1 = val
            print(mappings[val1])
            s = "static/output/" + mappings[val1]
            onlyfiles.append(mappings[val1])
            split = os.path.splitext(mappings[val1])[0]
            print(split)
            match = recipe_information.Image_Name[recipe_information.Image_Name == split].index.tolist()
            print(match)
            text = recipe_information['Instructions'][match[0]]
            recipe.append(text)
            print(text)
            completeName = os.path.join("static/output/", split + ".txt")
            with open(completeName, "w+") as f:
                f.writelines(text)
            f.close()
            cv2.imwrite(s, full_x[val])
            cv2.waitKey(0)
        print("\nRecipe0="+recipe[0])
        # files = os.listdir(mypath)
        # onlyfiles = list(filter(lambda x: '.jpg' in x, files))
        # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print(onlyfiles)
        return render_template('upload.html', filename=onlyfiles, query_filename = query_filename, recipe=recipe)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='output/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
