//import all the related spark libraries
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file
val data = sc.textFile("puppy_training_outcome.csv")
// "/FileStore/tables/a1ar0dlo1493966304172/puppy_training_outcome.csv"
// Split the data into training and test sets (20% held out for testing)
val trainingData = sc.parallelize(data.take((0.8*data.count).toInt))
val testData = sc.parallelize(data.take((0.2*data.count).toInt))
//Uncomment the following if you want to have random sampling of training data
//val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

//prepare the training data in (label, feature) form, where 'label' is the scalar outcome //matching 'feature' which may be multi-dimensional
val parsedTrainingData = trainingData.map { line =>
    val parts = line.split(',').map(_.toDouble)
    LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

//prepare the testing data in (label, feature) form, where 'label' is the scalar outcome //matching 'feature' which may be multi-dimensional
val parsedTestData = testData.map { line =>
    val parts = line.split(',').map(_.toDouble)
    LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

//Use training data to train and build a Naive Bayes Model
val model = NaiveBayes.train(parsedTrainingData, lambda = 1.0, modelType = "multinomial")

// Evaluate trained model on training data by forming (label, predicted label) pairs
val labelAndPredsTraining = parsedTrainingData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
}
// Evaluate trained model on testing data by forming (label, predicted label) pairs
val labelAndPredsTesting = parsedTestData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
}
//find the ratio of correct predciton to that of the entire dataset
val trainingError = labelAndPredsTraining.filter(r => r._1 == r._2).count.toDouble / labelAndPredsTraining.count
val testingError = labelAndPredsTesting.filter(r => r._1 == r._2).count.toDouble / labelAndPredsTesting.count
printf("Prediction reliability on trained data = %.2f%%\n", (100*trainingError))
printf("Prediction reliability on testing data = %.2f%%\n", (100*testingError))
//Mean Square Error(MSE) of trained model on testing data
val MSE = labelAndPredsTesting.map{ case(v, p) => math.pow((v - p), 2)}.mean()
printf("training Mean Squared Error for testing data: %f\n",MSE)
//Remind us what is the machine learning model used?
print(model)
