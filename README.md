# My Contributions
The initial data cloned from https://github.com/israksu/Hijja2/tree/master is organized by letter and in each folder there are folders of the letter posistions (ex. 2.1, meaning letter b at the beginning of a word). 
The dataset wasn't processed for CNN, so I had to create <make_cvs.ipynb> to store the image name and assign it with a value. I assigned each letter the same number the Hijja2 assigned (ex. letter alif is 1, ba is 2 etc). The <make_cvs.ipynb> references the inital way the data is organized, so you won't be able to run it with the current folder data. I will fix this problem when I am free.
Based on the tutorial I was following, the data needed to be divided by train, test, and validate datasets. The way I did it was not the most effeciet or linear, but also it took me a long time to figure out how the dataset is supposed to look like. In the meantime, I created <make_folder.ipynb> to store the images for testing, validating, and training in respective folders. I will also need to clean this up once I have more time.
From the random split of data, we also split the labels but I didn't do them together (because I didn't know), so I created a <make_label.ipynb> code. Moving forward this document can be merged with <make_folder.ipynb>. 
After I had stored my data and its labels correctly, I proceeded to run the model. Writing the code was an iterative process, with a lot of trial and error involved. There were challenges, particularly when dealing with data manipulation and ensuring the correct format and shape for the machine learning model. However, overcoming these difficulties was rewarding and a valuable learning experience. It was beneficial to delve into different aspects of Python, TensorFlow, and data preprocessing. Understanding the importance of appropriately preparing data for machine learning algorithms was a key takeaway from this task. Furthermore, debugging the code and resolving issues helped enhance problem-solving skills. The experience overall was enriching, and very educational. I have never worked with CNNs before so it was an infromative experience to say the least, espcially manipulating and preparing the data.

# Directory 
- <make_cvs.ipynb> : python code to convert the original dataset in all one cvs file with assigned values
- <make_folders.ipynb> : splits data into train, test, and validate, datasets
- <make_labels.ipynb> : I forgot to assign the split data the values, so I made this to assign them their values
- <train.ipynb> : I ran the models and tested the in this folder

# Models and Results 
The paper had an accuracy rate of 88% and their CNN model had 5 layers, 2 pooling and 3 convultional layers. My first 4 models have basic layering (two convo two pooling). I varied the base model with two optimzers and ways of fitting. I then tried to add more layers in models 5 and 6. 

![Alt text](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs00521-020-05070-8/MediaObjects/521_2020_5070_Fig5_HTML.png?as=webp)
## Models:
- model_1: used Adam optizer and ran for 10 epochs. I fit it with vaidation <model_1.fit(train_images, train_labs, validation_data=(val_images, val_labs), epochs=10)>.
- model_2: the same as the first but I fit it with 15 epochs and no validation <model_2.fit(train_images, train_labs, epochs=15)>
- model_3: model has the same number of layers but sgd optimzer and 15 epochs and no validation <model_3.fit(train_images, train_labs, epochs=15)>
- model_4: used sgd optimzer and 10 epochs with validation
- model_5: added an extra pooling and conv layers <layers.Conv2D(128, (3, 3), activation='relu', padding='same'))>
- model_6: added to model_5 also extra pooling and conv layers


## Results:
- model 1:
  
  214/214 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.2122 - loss: 3.2572

- model 2:
  
  214/214 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.7663 - loss: 0.8719
  
- model 3:

  214/214 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.0000e+00 - loss: nan
  
- model 4:

  214/214 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.2122 - loss: 3.2633
  
- model 5:

  214/214 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.7930 - loss: 0.7495
  
- model 6:

  214/214 ━━━━━━━━━━━━━━━━━━━━ 4s 16ms/step - accuracy: 0.8247 - loss: 0.7455

None of my results are better than the paper, however model 6 has 80% accuracy.There doesn't seem to be a big difference between the two optimzers for models 1 and 4, as they had the same accuracy rates. Model 3, however, had vauge results with <nan> loss. I am not sure why that is. 

# How to run the code

Please just run the train.ipynb, the other notebooks are for datamanipulation (unless you want to re-randomize the train-test-validation) 

# References 
My project is based off of the paper "Arabic Handwriting Recognition Using Convolutional Neural Network" by Najwa Altwajiry and Isra Al-Turaiki in 2020. The paper introduces the Hijja2 dataset, letters writtne by children in Riyadh's schools.
I followed the tutorial published by Medium and written by Abhishek Anand, (link https://medium.com/analytics-vidhya/image-text-recognition-738a368368f5)
To solve my bug problems, I used AI and the following links:

  https://stackoverflow.com/questions/71984678/unimplementederror-graph-execution-error-running-nn-on-tensorflow
  
  https://www.tensorflow.org/api_docs/python/tf/keras/Model
  
  https://discuss.pytorch.org/t/best-way-to-convert-a-list-to-a-tensor/59949
  
  https://stackoverflow.com/questions/65474081/valueerror-data-cardinality-is-ambiguous-make-sure-all-arrays-contain-the-same
  
