//import all the related spark libraries
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.tree.configuration.Algo._;
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel};
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

// Load in all Data
def load_file(filename: String) : org.apache.spark.rdd.RDD[String] = {
  val data = sc.textFile(filename);
  val header = data.first() // Extract Header
  data.filter(row => row != header) // Remove Header
}

val data_a = load_file("./data/puppy_info_feature_set_A.csv");
val data_b = load_file("./data/puppy_info_feature_set_B.csv");
val data_c = load_file("./data/puppy_info_feature_set_C.csv");
val data_d = load_file("./data/puppy_info_feature_set_D.csv");
val data_e = load_file("./data/puppy_info_feature_set_E.csv");
val data_f = load_file("./data/puppy_info_feature_set_F.csv");
val data_g = load_file("./data/puppy_info_feature_set_19.csv");
val data_h = load_file("./data/puppy_info_feature_set_24.csv");
val data_i = load_file("./data/puppy_info_feature_set_29.csv");
val data_j = load_file("./data/puppy_info_feature_set_34.csv");
val data_k = load_file("./data/puppy_info_feature_set_39.csv");
val data_l = load_file("./data/puppy_info_feature_set_42.csv");

// Train Function
def train_and_score_model(data: org.apache.spark.rdd.RDD[String], name: String, lambda: Double, model_type: String, split_random: Boolean) : Unit = {
  // Parse Data
  val parsed = data.map { line =>
    val parts = line.split(',').map(_.toDouble)
    LabeledPoint(parts(0), Vectors.dense(parts.tail))
  }

  // Split Data
  // If splitMode true, do randomly
  var splits = parsed.randomSplit(Array(0.8, 0.2), seed = 0);
  if(split_random == true) {
    splits = parsed.randomSplit(Array(0.8, 0.2));
  }
  val training = splits(0);
  val test = splits(1);

  // Train Naive Bayes
  val model = NaiveBayes.train(training, lambda = 1.0, modelType = model_type);

  // Score Model
  val train_predictionAndLabel = training.map(p => (model.predict(p.features), p.label));
  val test_predictionAndLabel = test.map(p => (model.predict(p.features), p.label));
  val train_accuracy = 100.0 * train_predictionAndLabel.filter(x => x._1 == x._2).count() / training.count();
  val test_accuracy  = 100.0 * test_predictionAndLabel.filter(x => x._1 == x._2).count() / test.count();

  // Print Accuracies
  println(f"$name with $model_type Naive Bayes with random $split_random: ($train_accuracy%.2f%%, $test_accuracy%.2f%%)");
}

/******************************************************************************
***** QUESTION 2A: Analyzing effect of feature set size on training
******************************************************************************/
train_and_score_model(data_a, "set A", 1.0, "multinomial", false);
train_and_score_model(data_b, "set B", 1.0, "multinomial", false);
train_and_score_model(data_c, "set C", 1.0, "multinomial", false);
train_and_score_model(data_d, "set D", 1.0, "multinomial", false);
train_and_score_model(data_e, "set E", 1.0, "multinomial", false);
train_and_score_model(data_f, "set F", 1.0, "multinomial", false);
train_and_score_model(data_g, "set G", 1.0, "multinomial", false);
train_and_score_model(data_h, "set H", 1.0, "multinomial", false);
train_and_score_model(data_i, "set I", 1.0, "multinomial", false);
train_and_score_model(data_j, "set J", 1.0, "multinomial", false);
train_and_score_model(data_k, "set K", 1.0, "multinomial", false);
train_and_score_model(data_l, "set L", 1.0, "multinomial", false);

train_and_score_model(data_a, "set A", 1.0, "bernoulli", false);
train_and_score_model(data_b, "set B", 1.0, "bernoulli", false);
train_and_score_model(data_c, "set C", 1.0, "bernoulli", false);
train_and_score_model(data_d, "set D", 1.0, "bernoulli", false);
train_and_score_model(data_e, "set E", 1.0, "bernoulli", false);
train_and_score_model(data_f, "set F", 1.0, "bernoulli", false);
train_and_score_model(data_g, "set G", 1.0, "bernoulli", false);
train_and_score_model(data_h, "set H", 1.0, "bernoulli", false);
train_and_score_model(data_i, "set I", 1.0, "bernoulli", false);
train_and_score_model(data_j, "set J", 1.0, "bernoulli", false);
train_and_score_model(data_k, "set K", 1.0, "bernoulli", false);
train_and_score_model(data_l, "set L", 1.0, "bernoulli", false);

/******************************************************************************
***** QUESTION 2B: Analyzing training consistency with random splits
******************************************************************************/
def run_x_times(data: org.apache.spark.rdd.RDD[String], name: String) {
  for(i <- 1 to 3) {
    train_and_score_model(data, name, 1.0, "multinomial", true);
  }
  for(i <- 1 to 3) {
    train_and_score_model(data, name, 1.0, "multinomial", false);
  }
}

run_x_times(data_a, "set A");
run_x_times(data_b, "set B");
run_x_times(data_c, "set C");
run_x_times(data_d, "set D");
run_x_times(data_e, "set E");
run_x_times(data_f, "set F");
run_x_times(data_g, "set G");
run_x_times(data_h, "set H");
run_x_times(data_i, "set I");
run_x_times(data_j, "set J");
run_x_times(data_k, "set K");
run_x_times(data_l, "set L");

/******************************************************************************
***** QUESTION 2C: Analyze varying lambda parameter's effect on training
******************************************************************************/
for(i <- 1 to 10) {
  printf("Lambda value: %.2f ", i*0.1)
  train_and_score_model(data_a, "set A", 0.1*i, "multinomial", false);
}
for(i <- 1 to 10) {
  printf("Lambda value: %.2f ", i*0.1)
  train_and_score_model(data_a, "set A", 0.1*i, "bernoulli", false);
}

/******************************************************************************
***** EXTRA: Comparing Naive Bayes Performance
******************************************************************************/
// Train Function on other models for comparing Naive Bayes performance
def train_and_score_other_models(data: org.apache.spark.rdd.RDD[String], name: String, split_random: Boolean) : Unit = {
  // Parse Data
  val parsed = data.map { line =>
    val parts = line.split(',').map(_.toDouble)
    LabeledPoint(parts(0), Vectors.dense(parts.tail))
  }

  // Split Data
  // If splitMode true, do randomly
  var splits = parsed.randomSplit(Array(0.8, 0.2), seed = 0);
  if(split_random == true) {
    splits = parsed.randomSplit(Array(0.8, 0.2));
  }
  val training = splits(0);
  val test = splits(1);

  // Decision Tree Classifier
  val numClasses = 2
  val categoricalFeaturesInfo = Map[Int, Int]()
  val impurity = "gini"
  val maxDepth = 10
  val maxBins = 32

  val dtmodel = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

  // Score Model
  var train_predictionAndLabel = training.map(p => (dtmodel.predict(p.features), p.label));
  var test_predictionAndLabel = test.map(p => (dtmodel.predict(p.features), p.label));
  var train_accuracy = 100.0 * train_predictionAndLabel.filter(x => x._1 == x._2).count() / training.count();
  var test_accuracy  = 100.0 * test_predictionAndLabel.filter(x => x._1 == x._2).count() / test.count();

  // Print Accuracies
  println(f"$name with Decision Tree with random $split_random: ($train_accuracy%.2f%%, $test_accuracy%.2f%%)");

  // Linear SVM Classifier
  val numIterations = 100
  val svmmodel = SVMWithSGD.train(training, numIterations)

  // Score Model
  train_predictionAndLabel = training.map(p => (svmmodel.predict(p.features), p.label));
  test_predictionAndLabel = test.map(p => (svmmodel.predict(p.features), p.label));
  train_accuracy = 100.0 * train_predictionAndLabel.filter(x => x._1 == x._2).count() / training.count();
  test_accuracy  = 100.0 * test_predictionAndLabel.filter(x => x._1 == x._2).count() / test.count();

  // Print Accuracies
  println(f"$name with Linear SVM with random $split_random: ($train_accuracy%.2f%%, $test_accuracy%.2f%%)");

}

train_and_score_other_models(data_a, "set A", false);
train_and_score_other_models(data_b, "set B", false);
train_and_score_other_models(data_c, "set C", false);
train_and_score_other_models(data_d, "set D", false);
train_and_score_other_models(data_e, "set E", false);
train_and_score_other_models(data_f, "set F", false);
train_and_score_other_models(data_g, "set G", false);
train_and_score_other_models(data_h, "set H", false);
train_and_score_other_models(data_i, "set I", false);
train_and_score_other_models(data_j, "set J", false);
train_and_score_other_models(data_k, "set K", false);
train_and_score_other_models(data_l, "set L", false);
