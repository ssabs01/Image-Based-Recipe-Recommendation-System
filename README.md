## Image Based Recipe Recommendation System

The aim of this project is to recommend recipes based on a query image of a dish that is provided as an input. This project involves following modules:
1. **Data Preprocessing** :  All the images in the food image database and the query image is reshaped to the size 224*169*3. 
2. **Feature Extraction** : Features are extracted using feature extraction techniques like KAZE, SIFT, SURF and HOG.
3. **Ensembling Features** : Here, all Features obtained from above feature extraction techniques are combined/stacked.
4. **Feature matching** : Ffeature matching is performed between the query image and the images of the predicted class using cosine similarity.
5. **Class Prediction** : Various classifiers are used to predict which class the query image belongs to.
7. **Image and Recipe Retrieval** : First 5 images and their recipes are retrieved based on the least distances or maximum similarities.
8. **UI** : These dishes and their recipes are displayed in a website using HTML and server is hosted using Flask.
<br>
<div align="center"><img src="Modules.png" height='600px'/></div>


