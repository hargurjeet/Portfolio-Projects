# Data Science Portfolio
Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form of iPython Notebooks markdown files.

_Note: Data used in the projects (accessed under data directory) is for demonstration purposes only._

- ### Machine Learning

	- [Abalone Dataset](https://github.com/hargurjeet/MachineLearning/blob/Abalone-Dataset/Abalone_Dataset_Analysis.ipynb): A regression model to predict the age of abalone on the given objective measures. 
	- [Bank Note Dateset](https://github.com/hargurjeet/MachineLearning/blob/Bank-Note-Dataset/Bank_Note_Analysis.ipynb): The Banknote Dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph.It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 1,372 observations with 4 input variables and 1 output variable.
	- [Swedish Auto Insurance Dataset](https://github.com/hargurjeet/MachineLearning/blob/Swedish-Auto-Insurance-Dataset/Swedish_Auto_Insurance_Dataset.ipynb): The Swedish Auto Insurance Dataset involves predicting the total payment for all claims in thousands of Swedish Kronor, given the total number of claims.It is a regression problem.The model performance of predicting the mean value is an RMSE of approximately 118 thousand Kronor.
	- [Wine Quality Dataset](https://github.com/hargurjeet/MachineLearning/blob/Wine-Quality-Dataset/Wine_Quality_Dataset.ipynb): The Wine Quality Dataset involves predicting the quality of white wines on a scale given chemical measures of each wine.
	- [Pima Indians Diabetes Dataset](https://github.com/hargurjeet/MachineLearning/blob/Pima-Indians-Diabetes-Dataset/Pima_Indians_Diabetes_Dataset.ipynb): The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 768 observations with 8 input variables and 1 output variable. Missing values are believed to be encoded with zero values.The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 65%. I achieved a classification accuracy of approximately 77%.
	- [Sonar Dataset](https://github.com/hargurjeet/MachineLearning/blob/Sonar-Dataset/Sonar_Dataset.ipynb): The Sonar Dataset involves the prediction of whether or not an object is a mine or a rock given the strength of sonar returns at different angles.It is a binary (2-class) classification problem. The number of observations for each class is not balanced.The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 53%. I achieved a classification accuracy of approximately 938%.
	- [Ionosphere Dataset](https://github.com/hargurjeet/MachineLearning/blob/Sonar-Dataset/Sonar_Dataset.ipynb): The Ionosphere Dataset requires the prediction of structure in the atmosphere given radar returns targeting free electrons in the ionosphere.There are 351 observations with 34 input variables and 1 output variable. I have achieved the classifcation accuracy of 93%.
	- [Wheat Seeds Dataset](https://github.com/hargurjeet/MachineLearning/blob/Wheat-Seeds/Wheat_Seeds_Analysis_Pytorch.ipynb): The Wheat Seeds Dataset involves the prediction of species given measurements of seeds from different varieties of wheat.
	- [Califonia Housing Dataset](https://github.com/hargurjeet/MachineLearning/blob/Califonia-Housing-Dataset/Califonia_Housing_Analysis.ipynb): The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. The object is to identify the median housing value in that area. This is a regression problem


	_Tools: scikit-learn, Pandas, Seaborn, Matplotlib, Pygame_ 
	
- ### DeepLearning

- #### PyTorch
  - [Wheat Seed Dataset](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/Wheat_Seeds_Analysis_Pytorch_blogs.ipynb): The datasets involve the prediction of species given measurements of seeds from different varieties of wheat. I build a logistic regression model and achieved an accuracy of **78% under 15 epochs**.
  - [CFAR-10](https://www.kaggle.com/c/cifar-10): The dataset contains over **60,000 images** belonging to 10 classes. I have developed the following neural networks to evaluate their performance.
    - Built [Feed Forward neural network(ANN)](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/CFAR_10_Dataset.ipynb) and achievied an accurcy of **48%**.
    - Built [Convolutional Neural Network(CNN)](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/CNN_CFAR_10_Dataset.ipynb) and improved the accuracy till **75%**.
    - Applied technique like Data normalization, Data augmentation, Batch normalization, Learning rate scheduling, Weight Decay, Gradient clipping...etc to further improve the model **accuracy to 90%**. You can access my notebook [here](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/CFAR_10_Image_Classifier.ipynb)
  - [Fruit 360 Classifier](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/Fruit_360_Classification.ipynb): The problem involves predicting the fruit class from a set of 131 classes with training data of over 65k images. I build a CNN neural network to achieve the highest possible accuracy(i.e. **98%) under 5 epochs** in less than **4 mins**.  
   


- #### Tensorflow
	- [GTSRB Dataset](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/GTRSB%20-%20CNN%20%28TensorFlow%29.ipynb): The dataset contains over **39k images** and over **40 classes**. I have to build a neural network with CNN architecture using Tensorflow and applying techniques like image augmentation to achieve accuracy of **85%**.

Project updates in progress

- ### Natural Language Processing
	- [Twitter Disaster Tweets](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/NLP_Twitter_Disaster_Tweets.ipynb): The dataset involves processing the text to predict if a tweet signifies disaster information or not. I build the model using the **NLTK lib** for text processing and TensorFlow for building up the neural network. I have applied **regular expression, stopwords, tokenization, pad sequences** to build the model and implemented it using **TensorFlow**

Project updates in progress
	_Tools: NLTK, scikit_

- ### Data Analysis and Visualisation
	- [Zomato Resturant Analysis](https://nbviewer.jupyter.org/github/hargurjeet/Data-Analysis-Using-Python/blob/main/Zomato%20Restaurant%20Analysis.ipynb): I really get  fascinated by good quality food being served in the restaurants and would like to help community find the best cuisines around their area like highly rated resturants locality wise, Cost Vs Rating etc.You can also access my blog [here](https://gurjeet333.medium.com/explanatory-data-analysis-of-zomato-restaurant-data-71ba8c3c7e5e)
  - [Olympics Dataset Analysis](https://nbviewer.jupyter.org/github/hargurjeet/Data-Analysis-Using-Python/blob/main/olympics_dataset_analysis.ipynb): This is a historical dataset on the modern Olympic Games, including all the Games from Athens 1896 to Rio 2016. In this notebook I use python to run some data exploration techniques to provid my view of viewing the dataset like understand the impact of Height,Weight and Age in winning the medals, Women participaiton over the years...etc.You can also access my blog [here](https://gurjeet333.medium.com/data-exploration-of-historical-olympics-dataset-2d50a7d0611d)
  - [Covid 19 Dataset](https://nbviewer.jupyter.org/github/hargurjeet/Data-Analysis-Using-Python/blob/main/Covid19-Analysis.ipynb): The below analysis is performed on Covid 19 Dataset which is freely avabliable on GitHub. I have tried performing analysis on various features to understand the spread of virus across various geographies and how the induvial countries have been impacted economically.My blog can be accessed [here](https://gurjeet333.medium.com/covid-19-explanatory-data-analysis-76cab46c48d1#ad87)
		
	_Tools: Pandas, Folium, Seaborn and Matplotlib,plotly_
- ### Data Science blogs on Medium 	
	
If you liked what you saw, want to have a chat with me about the portfolio, work opportunities, or collaboration, shoot an email at gurjeet333@gmail.com
