import os
from glob  import glob
import pandas as pd
from functools import reduce
from xml.etree import ElementTree as et
xmlfiles = glob('./annotations/*.xml')
replace_text = lambda x: x.replace('//','/')
xmlfiles=list (map(replace_text,xmlfiles))
def extract_text(filename):
    tree=et.parse(filename)
    root=tree.getroot()
    # extract filename
    image_name = root.find('filename').text
    #width and height of the image
    width = root.find('size').find('width').text
    height=root.find('size').find('height').text
    objs=root.findall('object')
    parser = []
    for obj in objs:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin= bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin =bndbox.find('ymin').text 
        ymax= bndbox.find('ymax').text
        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])
    return parser
parser_all = list(map(extract_text,xmlfiles))
data = reduce (lambda x, y: x+y, parser_all)
df = pd.DataFrame(data, columns = ['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])

# type converston
cols = ['width','height', 'xmin', 'xmax', 'ymin', 'ymax']
df[cols] = df[cols].astype(int)


# center x, center y
df['center_x'] = ((df['xmax']+df['xmin'])/2)/df['width']
df['center_y'] = ((df['ymax']+df['ymin'])/2)/df['height']

df['w'] = (df['xmax']-df['xmin'])/df['width']
# h
df['h'] = (df['ymax']-df['ymin'])/df['height']
# Label encoding
def label_encoding(x):
    labels = {'With Helmet':2, 'Without Helmet':3}
    return labels[x]
df['id']=df['name'].apply(label_encoding) 

cols = ['filename', 'id', 'center_x','center_y', 'w', 'h']
groupby_obj = df[cols].groupby('filename')
fol='F:\gsfc\project\helmet dectection and responce\Helmet Detection\Helmet Detection\\annotations'
def save_data(filename, folder_path, group_obj):
     # save the Labels
    text_filename = os.path.splitext(filename)[0]+'.txt'
    group_obj.get_group(filename).set_index('filename').to_csv(text_filename,sep=' ',index=False,header=False)
filename_series = pd.Series (groupby_obj.groups.keys())

filename_series.apply(save_data, args=(fol, groupby_obj))