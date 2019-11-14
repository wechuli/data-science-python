# Machine Learning

Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed. The examples that the system uses to learn are called the training set. Each training example is called a training instance(or sample).

Machine Learning is great for

- Problems for which existing solutions require a kit of fine-tuning or long list of rules
- Complex problems for which using a traditional approach yields no good solution: the best Machine Learning techniques can perhaps find a solution
- Fluctuating environments: a Machine Learning System can adapt to a new data
- Getting insights about complex problems

## Examples of Applications

- Analyzing images of products on a production line to automatically classify them
- Detecting tumors in brain scans
- Automatically classifying news articles
- Automatically flagging offensive comments on discussion forums
- Summarizing long documents automatically
- Creating a chatbot or a personal assistant
- Forecasting your company's revenue next year based on many performance metrics
- Making your app react to voice commands
- Detecting credit card fraud
- Segmenting clients based on their purchase so that you can design a different marketing strategy for each segment.
- Representing a complex, high-dimensional dataset in a clear and insightful diagram
- Recommending a product that a client may be interested in, based on past purchases
- Building an intelligent bot for a game

## Types of Machine Learning

### Supervised Vs Unsupervised Learning

- In supervised learning, the training set you feed to the algorithm includes the desired solutions. Important supervised learning algorithms include:
  - k-Nearest Neighbors
  - Linear Regression
  - Logistic Regression
  - Support Vector Machines (SVMs)
  - Decision Trees and Random Forests
  - Neural Nets
- Unsupervised learning is the trianing with unlabeled data.Examples of algorithms include
  - Clustering- K-Means,DBSCAN,Hierarchial Cluster Analysis(HCA)
  - Anomaly detection and novelty detection- One-class SVM,Isolation Forest
- Visualization and dimensionality reduction - Principal Component Analysis (PCA), Kernel PCA

It is often a good idea to try to reduce the dimension of your training data using dimensionality reduction algorithm before you feed it to another Machine Learning algorithm

#### Reinforcement Learning

Reinforcement Learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions and get rewards in return (or penalties in the form of negative rewards). It must then learn by itself what is the best strategy called a policy, to get the most reward over time. It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

### Batch and Online Learning

Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.

#### Batch learning

In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning.

If you want a batch learning system to know about new data, you need to train a new version of the system from scratch on the full dataset, then stop the old system and replace it with the new one.

#### Online learning

In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate.

### Instance-Based Versus Model-Based Learning

One more way to categorize Machine Learning systems is by how they generalize. Most Machine Learning tasks are about making predictions.

#### Instance-based learning

Possibly the most trivial form of learning is simply to learn by heart. The system learns the examples by heart, then generalizes to new case by using a similarity measure to compare them to the learned examples.

#### Model-based learning

Another way to generalize from a set of examples is to build a model of these examples and then use that model to make predictions. This is called model-based learning.

Here you need to specify a performance measure. You can either define a utility function (or fitness function) that measures how good your model is, or you can define a cost function that measures how bad it is. For Linear Regression problems, people typically use a cost function that measures the distance between the linear model’s predictions and the training examples; the objective is to minimize this distance.

## Main Challenges of Machine Learning

In short, since your main task is to select a learning algorithm and train it on some data, the two things that can go wrong are "bad algorithm" and "bad data"

### Insufficient Quantity of Training Data

It takes a lot of data for most Machine Learning algorithms to work properly.

### Nonrepresentative Training Data

In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to. It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have sampling nose (i.e. nonrepresentative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias.

### Poor-Quality Data
 Obviously, if your training data is full of errors, outliers and noise, it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well. It is often well worth the effort to spend time cleaning up your training data.

### Irrelevant Features

Your system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones. A critical part of the success of Machine Learning project is coming up with a good set of features to train on. This process, called feature engineering, involves the following steps:

- *Feature Selection* - selecting the most useful features to train on among existing features
- *Feature extraction* - Combining existing features to produce a more useful one, dimesnionality reduction algorithms can help
- Creating new features by gathering new data

### Overfitting the Training Data

This is where the model performs well on the training data, but does not generalize well. Overfitting happens when the model is too complex to the amount and noisiness of the training data. Here are possible solutions:
- Simplify the model by selecting one with fewer parameters
- Gather more training data
- Reduce the noise in the training data

Constraining a model to make is simpler and reduce the risk of overfitting is called regularization. You want to find the right balance between fitting the training data perfectly and keeping the model simple enough to ensure that it will generalize well.

The amount of regularization to apply during learning can be controlled by a *hyperparameter*. A hyperparameter is a parameter of a learning algorithm (not of the model). As such, it is not affected by the learning algorithm itself; it must be set prior to training  and remains constant during training.

### Underfitting the Training Data

Underfitting the model occurs when your model is too simple to learn the underlying structure of the data.

## Testing and Validating

The only way to know how well a model will generalize to new cases is to actually try it out on new cases. 

To do this, you split your data into two sets: training set and the test set. The error rate on new cases is called generalization error (or out-of-sample error), and by evaluating your model on the test set, you get an estimate of this error.

### Hyperparameter Tuning and Model Selection


The problem is that you measured the generalization error multiple times on the test set, and you adapted the model and hyperparameters to produce the best model for that particular set. This means that the model is unlikely to perform as well on new data.

A common solution to this problem is called holdout validation: you simply hold out part of the training set to evaluate several candidate models and select the best one. This new held-out set is called the validation set.More specifically, you train multiple models with various hyperparameters on the reduced training set (i.e., the full training set minus the validation set), and you select the model that performs best on the validation set. After this holdout validation process, you train the best model on the full training set (including the validation set), and this gives you the final model. Lastly, you evaluate this final model on the test set to get an estimate of the generalization error.

This solution usually works quite well. However, if the validation set is too small, then model evaluations will be imprecise: you may end up selecting a suboptimal model by mistake. Conversely, if the validation set is too large, then the remaining training set will be much smaller than the full training set. Why is this bad? Well, since the final model will be trained on the full training set, it is not ideal to compare candidate models trained on a much smaller training set. It would be like selecting the fastest sprinter to participate in a marathon. One way to solve this problem is to perform repeated cross-validation, using many small validation sets. Each model is evaluated once per validation set after it is trained on the rest of the data. By averaging out all the evaluations of a model, you get a much more accurate measure of its performance. There is a drawback, however: the training time is multiplied by the number of validation sets.

### Data Mismatch

In some cases, it's easy to get a large amount of data for training but this data probably won't perfectly be representative of the data that will be used in production.

### No Free Lunch Theorem

A model is a simplified version of the observations.The simplifications are meant to discard the superfluous details that are unlikely to generalize to new instances. To decide what data to discard and what data to keep, you must make assumptions.

        If you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.

There is no model that is a priori guaranteed to work better.

## Pipelines

A sequence of data processing components is called a data pipeline.  Pipelines are very common in Machine Learning systems, since there is a lot of data to manipulate and many data transformations to apply.

Components typically run asynchronously. Each component pulls in a large amount of data, processes it and spits out the result in another data store. Then, some time later, the next component in the pipeline pulls this data and spits out its own output. Each component is fairly self-contained: the interface between components is simply the data store. This makes the system simple to grasp and different teams can focus on different components. Moreover, if a component breaks down, the downstream components can often continue to run normally (at least for a while) by just using the last output from the broken component. This makes the architecture quite robust.

