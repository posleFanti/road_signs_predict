from sklearn.model_selection import train_test_split
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import pandas as pd
import shutil
import os
import cv2 as cv
import json
import yaml

label_map = pd.read_json('./rtsd-dataset/label_map.json', typ='series')
label_map = label_map.reset_index()
label_map.columns = ['label', 'class']

train_anno = pd.read_json('./rtsd-dataset/train_anno.json', typ='series')
train_anno_reduced = pd.read_json('./rtsd-dataset/train_anno_reduced.json', typ='series')
val_anno = pd.read_json('./rtsd-dataset/val_anno.json', typ='series')

df_images = pd.DataFrame(train_anno.images)
df_annotations = pd.DataFrame(train_anno.annotations)
df_categories = pd.DataFrame(train_anno.categories)

nclasses = df_annotations.category_id.nunique()

label_mapping = dict(zip(label_map['class'], label_map['label']))
cat_mapping = dict(zip(df_categories['id'], df_categories['name']))


def show_image_with_annotations(image_id):
    image_data = df_images[df_images['id'] == image_id].iloc[0]
    annotations = df_annotations[df_annotations['image_id'] == image_id]
    print(annotations)
    image_path = os.path.join("./rtsd-dataset/rtsd-frames", image_data['file_name'])
    image = cv.imread(image_path)

    fig, ax = plt.subplots(1)
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    for _, annotation in annotations.iterrows():
        bbox = annotation['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

#show_image_with_annotations(100)


def convert_bbox(bbox, width, height):
    x_center = (bbox[0] + bbox[2] / 2) / width
    y_center = (bbox[1] + bbox[3] / 2) / height
    w = bbox[2] / width
    h = bbox[3] / height
    return x_center, y_center, w, h


images_folder = "rtsd-frames"
output_folder = ["train", "valid", "test"]
output_subfolders = ["images", "labels"]

train_data, test_data = train_test_split(train_anno.images, test_size=0.2, random_state=42)
train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

def yolo_format_folders():
    for data, folder in zip([train_data, valid_data, test_data], output_folder):
        labels_dir = f'./rtsd-dataset/{folder}/labels'
        images_dir = f'./rtsd-dataset/{folder}/images'
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        for image_info in tqdm(data):
        
            image_id = image_info["id"]
            file_name = image_info["file_name"].split('/')[-1].split('.')[0]
            annotations = df_annotations[df_annotations.image_id==image_id].to_dict('records')
            labels_path = f'./rtsd-dataset/{folder}/labels/{file_name}.txt'

            with open(labels_path, "w") as label_file:
                for annotation in annotations:
                    cat_id = annotation["category_id"]
                
                    cls = cat_id-1
                
                    bbox = convert_bbox(annotation["bbox"], image_info["width"], image_info["height"])
                    label_file.write(f"{cls} {' '.join(map(str, bbox))}\n")

            image_path_src = r'./rtsd-dataset/rtsd-frames/rtsd-frames/'+file_name
            image_path_dst = '/'.join(['./rtsd-dataset', folder, 'images',file_name])
            shutil.copy(image_path_src+'.jpg', image_path_dst+'.jpg')

train_anno = pd.read_json('rtsd-dataset/train_anno.json', typ='series')
label_map = pd.DataFrame(train_anno.categories)
label_map['id'] = label_map['id'].astype('str')

del train_anno

signs = pd.read_excel('signs_names.xlsx')
label_map = label_map.merge(signs, left_on='name', right_on='Номер знака', how='left')
label_map = label_map.rename(columns={"Номер знака": "sign_id", "Название": "label"})
label_map.to_excel('signs_desc.xlsx')
label_dict = {i: name for i, name in enumerate(label_map['label'].values.tolist())}

dict_file = [
    {'names': label_map['label'].values.tolist()},
    {'ncc' : [155]},
    {'test' : ['/Users/alex/cv_kursach/rtsd-dataset/test']},
    {'train' : ['/Users/alex/cv_kursach/rtsd-dataset/train']},
    {'val' : ['/Users/alex/cv_kursach/rtsd-dataset/valid']}
]

with open(r'data_ru.yaml', 'w', encoding='utf-8') as file:
    documents = yaml.dump(dict_file, file, allow_unicode=True)

