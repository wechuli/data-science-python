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

