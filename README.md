# Heart-Disease-Prediction-using-Neural-Networks
Heart Disease Prediction using Neural Networks

This project uses a neural network model to predict the presence of heart disease based on various medical attributes. The model is trained using a dataset collected from the Cleveland Clinic Foundation, which is a commonly used subset of the UCI Heart Disease dataset.

 Dataset

The dataset contains 14 key attributes related to patient health, including:

- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Serum cholesterol  
- Fasting blood sugar  
- Resting electrocardiographic results  
- Maximum heart rate achieved  
- Exercise induced angina  
- ST depression induced by exercise  
- Slope of the peak exercise ST segment  
- Number of major vessels colored by fluoroscopy  
- Thalassemia  
- Target (presence or absence of heart disease)

We are using a cleaned version of the Heart Disease dataset, hosted on GitHub.
The dataset was originally from the UCI Machine Learning Repository.

Dataset URL: https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv

 Project Structure

1. *Importing the Dataset*  
   Load and explore the dataset using pandas.

2. *Preprocessing the Data*  
   Handle missing values, normalize data, and encode categorical variables.

3. *Splitting the Data*  
   Split the dataset into training and testing sets (80/20 split).

4. *Building the Neural Network*  
   Create a simple sequential model using Keras with input, hidden, and output layers.

5. *Training the Model*  
   Train the model using binary crossentropy loss and accuracy as the evaluation metric.

6. *Evaluating the Model*  
   Compare predicted vs. actual values and evaluate model performance using metrics.

 Libraries Used

- pandas  
- numpy  
- matplotlib  
- sklearn  
- tensorflow.keras

 Output Example
📊 Results

After training the neural network on the heart disease dataset, we tested it on a sample of data.

*Model Predictions:*  
[0. 1. 1. 0. 1.]

*True Labels:*  
[0 0 1 0 1]

As we can see, the model correctly predicted 4 out of 5 cases in this batch. This shows that the model is performing well on binary classification for heart disease prediction.


Author
This project was developed as a tutorial to understand how to apply deep learning in healthcare using Python and Keras.
