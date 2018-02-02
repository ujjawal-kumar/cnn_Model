
# coding: utf-8

# In[1]:


from coco2pascal import *


# In[2]:


dbpath = "D:/aditya/annotations/instances_train2014.json"
dst = r"D:\aditya\coco2014\VOC2014\train_ann" 
subset = r"D:\aditya\coco2014\train2014"


# In[3]:


annotations_Path = Path(dbpath)
print(annotations_Path)


# In[4]:


images_Path = Path(subset)
print(images_Path)


# In[5]:


categories , instances= get_instances(annotations_Path)


# In[6]:


print(categories)


# In[9]:


dst = Path(dst).expand()


# In[10]:


print(dst)


# In[11]:


for i, instance in enumerate(instances):
        instances[i]['category_id'] = categories[instance['category_id']]
        print(i)


# In[ ]:


for name, group in iteritems(groupby('file_name', instances)):
    img = imread(images_Path / name)
    if img.ndim == 3:
        out_name = rename(name)
        annotation = root('VOC2014', '{}.jpg'.format(out_name), 
                              group[0]['height'], group[0]['width'])
        for instance in group:
            annotation.append(instance_to_xml(instance))
            etree.ElementTree(annotation).write(dst / '{}.xml'.format(out_name))
            print (out_name)
    else:
            print (instance['file_name'])

