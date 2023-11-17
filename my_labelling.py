import Kmeans as km
import KNN as knn
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, Plot3DCloud, \
    visualize_k_means
import numpy as np
import matplotlib.pyplot as plt

def retrieval_by_color(images, labels, str_list):
    """
    images: dataset of images that are given to us
    labels: labels we receive by applying our k-means algorithm to those images
    str-list: color of the images we want to retrieve
    returns: list of images that match the color of the str-list parameter
    """
    image_list = []
    for i, l in zip(images, labels):
        for j in l:  # CHECKS EACH COLOR OF THE LABEL
            if j in str_list:
                image_list.append(i)
    return image_list


def get_color_accuracy(km_colors, gt):
    """
    km_colors: colors that our k-means algorithm returns
    gt: ground truth of the dataset
    returns: accuracy on the k-means algorithm
    """
    total_sum = 0
    for a, b in zip(km_colors, gt[2]):
        already_in = []
        inside = 0
        for color in a:
            if color not in already_in:
                already_in.append(color)
                if color in b:
                    inside += 1
        total_sum += (inside / len(already_in))
    return (total_sum / len(km_colors)) * 100


def knn_accuracy(pred, gt):
    """
    pred: our KNN' prediction
    gt: ground truth of the dataset
    returns: accuracy on the KNN algorithm
    """
    i = 0
    correct_sum = 0
    while i < len(pred):
        if pred[i] == gt[i]:
            correct_sum += 1
        i += 1
    percentage = (correct_sum / len(pred)) * 100
    return percentage


def askForInteger():
    correct = False
    num = 0
    while (not correct):
        try:
            num = int(input("Introduce an integer: "))
            correct = True
        except ValueError:
            print('Error, introduce an integer')

    return num


if __name__ == '__main__':

    # Load all the images and Ground Truth
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
    test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended Ground Truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()

    cropped_images = crop_images(imgs, upper, lower)

    my_gt = read_extended_dataset(root_folder='./images/', extended_gt_json='./images/gt_reduced.json', w=60, h=80)

    quit = False
    option = 0

    while not quit:
        print("****************************************************************************************")
        print("1. Visualize k-means: compare the k-means result to the original image (we reduced the dataset to the "
              "first 5 images to make it more dynamic, "
              "you can change the value of the retrieved images in the option 1 on the code)")
        print("2. Retrieve all the images that matches the parameter' color you introduce by parameters (you can "
              "change it on the option 2, by default is 'Black')")
        print("3. Calculate k-means algorithm accuracy. You can change the threshold value on the "
              "tolerance value in _init_options_ (by default is 0.2), you can also change the km_init value (by "
              "default 'optimum')")
        print("4. Here you can see the KNN algorithm accuracy. You can change the distance used in the k_neighbours "
              "function, inside the cdist function (by default 'euclidean', others: Hamming, Minkowski, "
              "Sokalmichener, Russellrao)")
        print("5. Quit")
        print("Choose an option")
        print("****************************************************************************************")

        option = askForInteger()

        if option == 1:
            for im in imgs[0:5]:
                my_km = km.KMeans(im)
                my_km._init_centroids()
                my_km.get_labels()
                my_km.get_centroids()
                visualize_k_means(my_km, im.shape)
        elif option == 2:
            colors = []
            for i in cropped_images:
                my_km = km.KMeans(i)
                my_km._init_centroids()
                my_km.get_labels()
                my_km.get_centroids()
                new_color = km.get_colors(my_km.centroids)
                colors.append(new_color)
            selected_images = retrieval_by_color(imgs, colors, 'Black')
            visualize_retrieval(selected_images, 15)
        elif option == 3:
            colors = []
            for i in cropped_images:
                my_km = km.KMeans(i)
                my_km._init_centroids()
                my_km.get_labels()
                my_km.get_centroids()
                c = km.get_colors(my_km.centroids)
                colors.append(c)
            color_accuracy_percentage = get_color_accuracy(colors, my_gt)
            print(color_accuracy_percentage)
        elif option == 4:
            test_imgs = test_imgs[:, :, :, 0]
            train_imgs = train_imgs[:, :, :, 0]

            my_knn = knn.KNN(train_imgs, train_class_labels)
            preds = my_knn.predict(test_imgs, 10)
            accuracy = knn_accuracy(preds, test_class_labels)
            print(accuracy)
        elif option == 5:
            quit = True
        else:
            print("Introduce a number between 1 and 4")

    print("Fin")
