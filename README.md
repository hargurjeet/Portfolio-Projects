# Data Science Portfolio Projects
![](Images/datascience.jpeg)
Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form of iPython Notebooks markdown files.

- ### Data Analysis and Visualisation 
	- Zomato Resturant Analysis ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/Data-Analysis-Using-Python/blob/main/Zomato%20Restaurant%20Analysis.ipynb), [<img src="https://img.icons8.com/office/40/000000/blog.png"/>](https://blog.jovian.ai/explanatory-data-analysis-of-zomato-restaurant-data-71ba8c3c7e5e)):
		- Analyzed over 9000 restaurants with over 20 features.
		- Performed data analysis using Python(pandas, numpy) and building visualizations using matplot lib and seaborn.
		- Identified best ‘Breakfast’, ‘Fast Food’ and ‘Ice Cream’ parlours in various localities.
	
  - Olympics Dataset Analysis ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/Data-Analysis-Using-Python/blob/main/olympics_dataset_analysis.ipynb), [<img src="https://img.icons8.com/office/40/000000/blog.png"/>](https://medium.com/nerd-for-tech/data-exploration-of-historical-olympics-dataset-2d50a7d0611d)):
	  - Analyzed Olympics data from Athens 1896 to Rio 2016. The dataset contain over 2,70,000 records acorss 15 features. 
	  - I use python to run some data exploration techniques to provid my view of viewing the dataset like understand the impact of Height,Weight and Age in winning the medals, Women participaiton over the years...etc.
	  - Build intractive visuals using plotly. Also use seaborn and matplotlib for building visual.
	  - Perfomed analysis over best and worst performing counteris.
	  - Used python librairies seaborn and matplot lib.
	
  - Covid 19 Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/Data-Analysis-Using-Python/blob/main/Covid19-Analysis.ipynb), [<img src="https://img.icons8.com/office/40/000000/blog.png"/>](https://medium.com/geekculture/covid-19-explanatory-data-analysis-76cab46c48d1)): 
  	- The analysis was perfomred over 47,000 records of daily covid data across 7 continents. The analysis was performed on 40 features.
  	- Libraries used - Matplotlib, Seaborn, plotly and Pandas.
  	- Analysed the trend of worst affected conteries and conturies having lowest death rate.
  	- Econnomic impact of Covid 19 worldwide.
  	
	
- ### Machine Learning
	- Car Quality Detection ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/master/Used_Car_Quality_Detection.ipynb#top), [<img src="https://img.icons8.com/office/40/000000/blog.png"/>](https://blog.jovian.ai/machine-learning-with-python-implementing-xgboost-and-random-forest-fd51fa4f9f4c)): 
	
		- Problem Statement: One of the biggest challenges of an auto dealership purchasing a used car at an auto auction is the risk of that the vehicle might have serious issues that prevent it from being sold to customers.The challenge of this competition is to predict if the car purchased at the Auction is a Kick (bad buy).
		- Processed data over 72k records with over 30 features to predict the quality of a car.
		- Libraries used - pandas, numpy, sklearn, matplotlib and seaborn.
		- Machine learning models implement - Random Forest, XGBoost.
		- Performed hyperparameter tuning along with random search CV to achieve accuracy of 88%.
		- Submitted this model to Kaggle Competetion scoring in top 10 percent at the leaderbord. 


	- Califonia Housing Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Califonia-Housing-Dataset/Califonia_Housing_Analysis.ipynb))
		- The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. The object is to identify the median housing value in that area.
		- The dataset have over 20,000 records and 9 features. 
		- Libraries used - numpy, OS, requests, urllib, Pandas, sklearn.
		- Feature analysis, stratified shuffle split, Visualized data to gain insights.
		- Data cleaning and preprocessing acivities - duplicate check, null values, One-Hot encoding, Feature scaling.
		- Model implement - Linear regression, Decision tree.
		- Hyperparameter tuning using gridsearchCV to evalute best model. RMSE of **47362** is achivied on test set.
		
	- Wine Quality Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Wine-Quality-Dataset/Wine_Quality_Dataset.ipynb))
		- Problem Statement - The Wine Quality Dataset involves predicting the quality of white wines on a scale given chemical measures of each wine.
		- The dataset have 5000 obseravation and 10 features.
		- Libarary used - Numpy, Pandas, Matplotlib and sklearn.
		- Feature analysis, Identiying relevant features, co relation of features with the target feature.
		- LinearRegression implemented and RMSE score of **0.75** is achivied.

	- Bank Note Dateset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Bank-Note-Dataset/Bank_Note_Analysis.ipynb))
		- Problem Statement - The Banknote Dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph.
		- The dataset have 1300 observeration of various noteparametes as features. It is a binary classification problem.
		- Data cleaning, Feature analysis and visuliazation using Pandas.
		- ML models implemented - Logistic regression, KNeighborsClassifier and SVM.
		- Hyperparamter tuning using GridsearchCV.
		- Model evluation - Precision and Recall calculated along with f1 scores.


	- Abalone Dataset ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Abalone-Dataset/Abalone_Dataset_Analysis.ipynb))
		- Business case - Predicting the age of abalone on the given physical measures. 
		- The dataset have over 4000 observation along with 8 features.
		- Build pipeline, implemented StandardScaler and One-Hote encoding for numberical and categorical columns.
		- Model Implemented - Linear regression, Decision Tree and Random forest. Evluation matrix - RMSE score.
		- Hyperparamter tuning using grid search CV and achieved an RMSE score of **2.254**.

	- Pima Indians Diabetes Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Pima-Indians-Diabetes-Dataset/Pima_Indians_Diabetes_Dataset.ipynb))
		- The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.
		- It is a binary (2-class) classification problem. There are 768 observations with 8 input variables and 1 output variable.
		- Libarary used - Numpy, Pandas, Matplotlib and sklearn. 
		- Implemented KNN classification. Parameter tuning using GridsearchCV.
		- The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 65%. I achieved a classification accuracy of approximately 77%.

	- Swedish Auto Insurance Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Swedish-Auto-Insurance-Dataset/Swedish_Auto_Insurance_Dataset.ipynb))
		- Problem Statement - The Swedish Auto Insurance Dataset involves predicting the total payment for all claims in thousands of Swedish Kronor.
		- Libarary used - Numpy, Pandas, Matplotlib and sklearn.
		- Folowing ML model implemented and evaluated against RMSE, MAE scores - Linear Regression, Decison trees, Random Forest
		- It is a regression problem.The model performance of predicting the mean value is an RMSE of approximately 118 thousand Kronor.

	- Ionosphere Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://github.com/hargurjeet/MachineLearning/blob/Ionosphere/Ionosphere_Data_Analysis.ipynb))
		- Problem Statement - The Ionosphere Dataset requires the prediction of structure in the atmosphere given radar returns targeting free electrons in the ionosphere.
		-  There are 351 observations with 34 input variables and 1 output variable.
		-  As the dataset beeing small, I implemented the k fold cross validations.
		-  ML models implemented - Logistic Regression,  KNeighborsClassifier, DecisionTreeClassifier, SVM)
		-  Have achieved the classification accuracy of 93%.

	- Sonar Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Sonar-Dataset/Sonar_Dataset.ipynb))
		- The Sonar Dataset involves the prediction of whether or not an object is a mine or a rock given the strength of sonar returns at different angles.
		- It is a binary (2-class) classification problem with 200 observations and 61 features.
		- ML Models implemented - LogisticRegression, LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier, SVM.
		- Hyperparameter tuning and achieved a classification accuracy of approximately 93%.

	- Wheat Seeds Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Wheat-Seeds/Wheat_Seeds_Analysis_Pytorch.ipynb))
		- The Wheat Seeds Dataset involves the prediction of species given measurements of seeds from different varieties of wheat.
		- There are 199 observations with 7 input variables and 1 output variable.
		- Implemented a Feed forward neural network.
		- Accuray of 60% is achivied.

	_Tools: scikit-learn, Pandas, Seaborn, Matplotlib, NumPy, Plotly_ 
	
- ### DeepLearning

- #### PyTorch 

  - [Fruit 360 Classifier](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/Fruit_360_Classification.ipynb): 
	  - Built a CNN and trained over 65k fruit images to predict the class of fruit from a set of 131 classes.
	  - Model is built on PyTorch along with the implementation of techniques like Data augmentation, Batch normalization, Learning rate scheduling, Weight Decay, Gradient clipping to achieve the best results.
	  - The tensors are trained and evaluated on GPU using PyTorch built-in CUDA library to build the model.
	  - Inference - Achieved model accuracy of 98% under 5 mins undermining the power of all techniques.	

  - [CFAR-10](https://www.kaggle.com/c/cifar-10): The dataset contains over **60,000 images** belonging to 10 classes. I have developed the following neural networks to evaluate their performance.
    - Built [Feed Forward neural network(ANN)](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/CFAR_10_Dataset.ipynb) and achievied an accurcy of **48%**.
    - Built [Convolutional Neural Network(CNN)](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/CNN_CFAR_10_Dataset.ipynb) and improved the accuracy till **75%**.
    - Applied technique like Data normalization, Data augmentation, Batch normalization, Learning rate scheduling, Weight Decay, Gradient clipping...etc to further improve the model **accuracy to 90%**. You can access my notebook [here](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/CFAR_10_Image_Classifier.ipynb).
  - [Wheat Seed Dataset](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/Wheat_Seeds_Analysis_Pytorch_blogs.ipynb): The datasets involve the prediction of species given measurements of seeds from different varieties of wheat. I build a logistic regression model and achieved an accuracy of **78% under 15 epochs**.
  
   


- #### Tensorflow
	- [GTSRB Dataset](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/GTRSB%20-%20CNN%20%28TensorFlow%29.ipynb): The dataset contains over **39k images** and over **40 classes**. I have to build a neural network with CNN architecture using Tensorflow and applying techniques like image augmentation to achieve accuracy of **85%**.
	- [Telco Customer Churn]():
  	- [CNN Model - Age, Gender, Ethnicity]():

Project updates in progress

- ### Natural Language Processing
	- [Twitter Disaster Tweets](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/NLP_Twitter_Disaster_Tweets.ipynb): The dataset involves processing the text to predict if a tweet signifies disaster information or not. I build the model using the **NLTK lib** for text processing and TensorFlow for building up the neural network. I have applied **regular expression, stopwords, tokenization, pad sequences** to build the model and implemented it using **TensorFlow**.
	- [NLP — Detecting Fake News On Social Media](https://nbviewer.jupyter.org/github/hargurjeet/DeepLearning/blob/main/Fake_News_Classifer.ipynb): The program help in identifying news articles programmatically if a news article is Fake or Not. I have tried comparing the techniques - **Bag of words** and **TF IDF** while approching to solve the problem and compared their accuracies.

Project updates in progress
	_Tools: NLTK, scikit_
	
- ### Recommedation-Systems
	- [Anime Recommendation](https://nbviewer.jupyter.org/github/hargurjeet/Recommedation-Systems/blob/main/Anime_Recommendation_Item_Based_CosineSimilarity.ipynb): This data set contains information on user preference data from 73,516 users on 12,294 anime. I have implemented Cosine similarity and Nearest neighbors machine learning algorithm to make recommendations to 6 nearest neighbors. I have used the item based Collborative technique to recommend anime's to users. For non technical users you can access by blog [here](https://gurjeet333.medium.com/building-recommendations-system-a-beginner-guide-8593f205bc0a).
	- [Book Recommendation](https://nbviewer.jupyter.org/github/hargurjeet/Recommedation-Systems/blob/main/Books_Recommendations.ipynb): The Dataset contatins over 10,000 user review. I have build a recommendation system using the weighted average technique. The jist of the analysis can also be accessed [here](https://gurjeet333.medium.com/what-should-i-read-next-books-recommendation-311666254817)	

- ### Large Language Models
	- [ATS Resume Expert](https://github.com/hargurjeet/ATS_Resume_Expert/tree/main): ATS_Resume_Expert is a data science application powered by Google Gemini Pro APIs and built using LangChain. It compares the job description of a job application against your resume and provides a score of match percentage. It also provides suggestions to improve your resume on keywords and areas.
 	- [MulitplePDFReader](https://github.com/hargurjeet/mulitplePDFReader): The Gemini Pro Streamlit App is a powerful tool that leverages the capabilities of Google's latest large language model, Gemini Pro, to provide users with comprehensive answers to their questions based on information extracted from multiple PDF documents.
 	 - [PDF info Retertiver with Pinecone](https://github.com/hargurjeet/pinecone_vectorDB): This repository contains the code for a data science project that develops an app to take a PDF document as input, process its details, and save the information as embeddings in the Pinecone vector database. The information is then retrieved by the OpenAI GPT-3 language model, specifically the Davinci model, based on the query asked by the user.
   	- [OCR-Based Invoice Data Extraction](https://github.com/hargurjeet/gemini_Image_Invoice_Info_Extraction): This is a data science project that aims to extract information from images of invoices and answer user queries. The application is capable of processing invoices in different languages and is powered by Google's latest LLM Gemini pro vision. The app was built using the Streamlit web interface.

- ### Data Science blogs on Medium 
  - [Machine Learning with Python: Implementing XGBoost and Random Forest](https://gurjeet333.medium.com/machine-learning-with-python-implementing-xgboost-and-random-forest-fd51fa4f9f4c)
  - [Exploratory Data Analysis of Zomato Restaurant data](https://blog.jovian.ai/explanatory-data-analysis-of-zomato-restaurant-data-71ba8c3c7e5e)
  - [Training Feed Forward Neural Network(FFNN) on GPU — Beginners Guide](https://medium.com/mlearning-ai/training-feed-forward-neural-network-ffnn-on-gpu-beginners-guide-2d04254deca9)
  - [7 Best Techniques To Improve The Accuracy of CNN W/O Overfitting](https://medium.com/mlearning-ai/7-best-techniques-to-improve-the-accuracy-of-cnn-w-o-overfitting-6db06467182f)
  - [Training Convolutional Neural Network(ConvNet/CNN) on GPU From Scratch](https://medium.com/mlearning-ai/training-convolutional-neural-network-convnet-cnn-on-gpu-from-scratch-439e9fdc13a5)
  - [Training Feed Forward Neural Network(FFNN) on GPU — Beginners Guide](https://medium.com/mlearning-ai/training-feed-forward-neural-network-ffnn-on-gpu-beginners-guide-2d04254deca9)
  - [Logistic Regression With PyTorch — A Beginner Guide](https://medium.com/analytics-vidhya/logistic-regression-with-pytorch-a-beginner-guide-33c2266ad129)
  - [PyTorch - Training Fruit 360 Classifier Under 5 mins](https://medium.com/geekculture/pytorch-training-fruit-360-classifier-under-5-mins-23153b46ec88)
  - [Deep Learning for Beginners Using TensorFlow](https://gurjeet333.medium.com/cnn-german-traffic-signal-recognition-benchmarking-using-tensorflow-accuracy-80-d069b7996082)
  - [Fake or Not ? Twitter Disaster Tweets](https://gurjeet333.medium.com/fake-or-not-twitter-disaster-tweets-f1a6b2311be9)
  - [NLP — Detecting Fake News On Social Media](https://medium.com/mlearning-ai/nlp-detecting-fake-news-on-social-media-aa53ff74f2ff)
  - [Building Recommendations System? A Beginner Guide](https://medium.com/mlearning-ai/building-recommendations-system-a-beginner-guide-8593f205bc0a)	
	
If you liked what you saw, want to have a chat with me about the portfolio, work opportunities, or collaboration, shoot an email at gurjeet333@gmail.com
