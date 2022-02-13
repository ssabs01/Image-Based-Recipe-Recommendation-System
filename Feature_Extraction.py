import os
import cv2
import pickle
import numpy as np
import plotly.express as px
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern

images_path = 'dataset\Food Images'


def read_images(datapath):
    labels = ['Biscuit', 'Brownie', 'Burger', 'Cake', 'Cookie', 'Cupcakes', 'Drinks', 'Ice Cream', 'Pasta', 'Pie',
              'Pizza', 'Salad', 'Salsa', 'Sandwich', 'Soup']
    mapping = {'Biscuit': 0, 'Brownie': 1, 'Burger': 2, 'Cake': 3, 'Cookie': 4, 'Cupcakes': 5, 'Drinks': 6,
               'Ice Cream': 7, 'Pasta': 8, 'Pie': 9, 'Pizza': 10, 'Salad': 11, 'Salsa': 12, 'Sandwich': 13, 'Soup': 14}
    images = []
    Imglabels = []
    num1 = 224
    num2 = 169
    count = 0
    dict_mapping = {}
    for label in labels:
        path = os.path.join(datapath, label)
        for img in os.listdir(path):
            dict_mapping[count] = img
            print(os.path.join(path, img))
            img = cv2.imread(os.path.join(path, img))
            new_img = cv2.resize(img, (num2, num1))
            images.append(new_img)
            Imglabels.append(mapping[label])
            count += 1
    return np.array(images), np.array(Imglabels), dict_mapping


def get_hog(images):
    result = np.array([hog(img, block_norm='L2') for img in images])

    return result


def get_sift(images):
    # SIFT descriptor for 1 image
    def get_image_sift(image, vector_size=15):
        alg = cv2.xfeatures2d.SIFT_create()
        kps = alg.detect(image, None)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

        # Making descriptor of same size
        # Descriptor vector size is 128
        needed_size = (vector_size * 128)
        if len(kps) == 0:
            return np.zeros(needed_size)

        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        if dsc.size < needed_size:
            # if we have less than 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

        return dsc

    # SIFT descriptor for all images
    features = []
    for i, img in enumerate(images):
        dsc = get_image_sift(img)
        features.append(dsc)

    result = np.array(features)

    return result


def get_kaze(images):
    # KAZE descriptor for 1 image
    def get_image_kaze(image, vector_size=32):
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if len(kps) == 0:
            return np.zeros(needed_size)

        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()

        if dsc.size < needed_size:
            # if we have less than 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        return dsc

    # KAZE descriptor for all images
    features = []
    for i, img in enumerate(images):
        dsc = get_image_kaze(img)
        features.append(dsc)

    result = np.array(features)

    return result


def combine_features(features, horizontal=True):
    """
    Array of features [f1, f2, f3] where each fi is a feature set
    eg. f1=rgb_flat, f2=SIFT, etc.
    """
    if horizontal:
        return np.hstack(features)
    else:
        return np.vstack(features)


def norm_features_minmax(train):
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train = min_max_scaler.fit_transform(train)

    return norm_train


def get_lbp(images):
    result = np.array([local_binary_pattern(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 10, 3).flatten() for img in images])

    return result


if __name__ == "__main__":
    full_x, full_y, mappings = read_images(images_path)
    print(full_x.shape)
    print(full_y.shape)
    with open('Mappings_folder.pickle', 'wb') as handle:
        pickle.dump(mappings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save('Paris_X_train', full_x)
    np.save('Paris_label', full_y)

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

    # HOG Features
    hog_features = get_hog(full_x)
    # print(hog_features.shape)
    np.save('Paris_HOG', hog_features)

    lbp_features = get_lbp(full_x)
    # lbp_features = np.load('Paris_LBP.npy')
    print(lbp_features.shape)
    np.save('Paris_LBP', lbp_features)

    sift_features = get_sift(full_x)
    # sift_features = np.load('Paris_SIFT.npy')
    print(sift_features.shape)
    np.save('Paris_SIFT', sift_features)

    kaze_features = get_kaze(full_x)
    # kaze_features = np.load('Paris_KAZE.npy')
    print(kaze_features.shape)
    np.save('Paris_KAZE', kaze_features)

    # Normalization
    hog_features = np.load('Paris_HOG.npy')
    norm_hog_features = norm_features_minmax(hog_features)
    print(norm_hog_features.shape)
    np.save('Paris_norm_HOG', norm_hog_features)

    hog_features = np.load('Caltech_HOG.npy')
    norm_hog_features = norm_features_minmax(hog_features)
    print(norm_hog_features.shape)
    np.save('Caltech_norm_HOG', norm_hog_features)

    lbp_features = np.load('Paris_LBP.npy')
    norm_lbp_features = norm_features_minmax(lbp_features)
    print(norm_lbp_features.shape)
    np.save('Paris_norm_LBP', norm_lbp_features)

    sift_features = np.load('Paris_SIFT.npy')
    norm_sift_features = norm_features_minmax(sift_features)
    print(norm_sift_features.shape)
    np.save('Paris_norm_SIFT', norm_sift_features)

    kaze_features = np.load('Paris_KAZE.npy')
    norm_kaze_features = norm_features_minmax(kaze_features)
    print(norm_kaze_features.shape)
    np.save('Paris_norm_KAZE', norm_kaze_features)

    # PCA
    hog_norm_features = np.load('Paris_norm_HOG.npy')
    pca = PCA(n_components=1400)
    pca_hog_features = pca.fit_transform(hog_norm_features)
    np.save('Paris_HOG_PCA', pca_hog_features)
    pca_hog_features = np.load('Paris_HOG_PCA.npy')
    print(pca_hog_features.shape)

    sift_norm_features = np.load('Paris_norm_SIFT.npy')
    print(sift_norm_features.shape)
    pca = PCA(n_components=503)
    pca_sift_features = pca.fit_transform(sift_norm_features)
    np.save('Paris_SIFT_PCA', pca_sift_features)
    pca_sift_features = np.load('Paris_SIFT_PCA.npy')
    print(pca_sift_features.shape)

    kaze_norm_features = np.load('Paris_norm_KAZE.npy')
    print(kaze_norm_features.shape)
    pca = PCA(n_components=600)
    pca_kaze_features = pca.fit_transform(kaze_norm_features)
    np.save('Paris_KAZE_PCA', pca_kaze_features)
    pca_kaze_features = np.load('Paris_KAZE_PCA.npy')
    print(pca_kaze_features.shape)

    lbp_norm_features = np.load('Paris_norm_LBP.npy')
    print(lbp_norm_features.shape)
    pca = PCA(n_components=3000)
    pca_lbp_features = pca.fit_transform(lbp_norm_features)
    np.save('Paris_LBP_PCA', pca_lbp_features)
    pca_lbp_features = np.load('Paris_LBP_PCA.npy')
    print(pca_lbp_features.shape)

    # Check Variance from PCA
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )
    fig.show()

    pca_surf_features = np.load('Paris_SURF_PCA.npy')
    # print(pca_surf_features.shape)

    # Combine PCA Features
    features_pca_2653 = None

    for t in (pca_hog_features, pca_sift_features, pca_kaze_features, pca_surf_features):
        if features_pca_2653 is None:
            features_pca_2653 = t
        else:
            features_pca_2653 = combine_features([features_pca_2653, t])

    print(features_pca_2653.shape)
    np.save('Paris_combined_2653', features_pca_2653)

    surf_norm_features = np.load('Paris_norm_SURF.npy')

    # print(labels.shape)

    # LDA Features
    lda = LDA()
    lda_features = lda.fit_transform(surf_norm_features, labels)
    np.save('Paris_HOG_LDA', lda_features)
    print(lda_features.shape)

    lda = LDA()
    lda_features = lda.fit_transform(sift_norm_features, labels)
    np.save('Paris_SIFT_LDA', lda_features)
    print(lda_features.shape)

    lda = LDA()
    lda_features = lda.fit_transform(surf_norm_features, labels)
    np.save('Paris_SURF_LDA', lda_features)
    print(lda_features.shape)

    lda = LDA()
    lda_features = lda.fit_transform(kaze_norm_features, labels)
    np.save('Paris_KAZE_LDA', lda_features)
    print(lda_features.shape)

    lda_hog_features = np.load('Paris_HOG_LDA.npy')
    lda_sift_features = np.load('Paris_SIFT_LDA.npy')
    lda_kaze_features = np.load('Paris_KAZE_LDA.npy')
    lda_surf_features = np.load('Paris_SURF_LDA.npy')

    # Combine LDA Features
    features_lda = None
    for t in (lda_hog_features, lda_sift_features, lda_kaze_features, lda_surf_features):
        if features_lda is None:
            features_lda = t
        else:
            features_lda = combine_features([features_lda, t])

    print(features_lda.shape)
    np.save('Paris_combined_LDA', features_lda)
