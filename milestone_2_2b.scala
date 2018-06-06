//import all the related spark libraries
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.tree.configuration.Algo._;
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel};

// Load in all Data
def load_file(filename: String) : org.apache.spark.rdd.RDD[String] = {
  val data = sc.textFile(filename);
  val header = data.first() // Extract Header
  data.filter(row => row != header) // Remove Header
}

val data_a = load_file("puppy_info_feature_set_A.csv");
val data_b = load_file("puppy_info_feature_set_B.csv");
val data_c = load_file("puppy_info_feature_set_C.csv");
val data_d = load_file("puppy_info_feature_set_D.csv");
val data_e = load_file("puppy_info_feature_set_E.csv");
val data_f = load_file("puppy_info_feature_set_F.csv");

// Parse Data into (Label, Features)
def parse_data(data: org.apache.spark.rdd.RDD[String]) : org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint] = {
  data.map { line =>
    val parts = line.split(',').map(_.toDouble)
    LabeledPoint(parts(0), Vectors.dense(parts.tail))
  }
}

// Train Function
def train_and_score_model_fixed(data: org.apache.spark.rdd.RDD[String], name: String) : Unit = {
  // Parse and Split Data
  val parsed = parse_data(data);
  val splits = parsed.randomSplit(Array(0.8, 0.2), seed = 0);
  val training = splits(0);
  val test = splits(1);

  // Train Naive Bayes
  val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial");

  // Score Model
  val train_predictionAndLabel = training.map(p => (model.predict(p.features), p.label));
  val test_predictionAndLabel = test.map(p => (model.predict(p.features), p.label));
  val train_accuracy = 1.0 * train_predictionAndLabel.filter(x => x._1 == x._2).count() / training.count();
  val test_accuracy = 1.0 * test_predictionAndLabel.filter(x => x._1 == x._2).count() / test.count();

  // Print Accuracies
  println("Accuracy of %s:", name);
  println(train_accuracy);
  println(test_accuracy);
}

train_and_score_model(data_a, "set A");
train_and_score_model(data_b, "set B");
train_and_score_model(data_c, "set C");
train_and_score_model(data_d, "set D");
train_and_score_model(data_e, "set E");
train_and_score_model(data_f, "set F");


def train_and_score_model_random(data: org.apache.spark.rdd.RDD[String], name: String) : Unit = {
  // Parse and Split Data
  val parsed = parse_data(data);
  val splits = parsed.randomSplit(Array(0.8, 0.2));
  val training = splits(0);
  val test = splits(1);

  // Train Naive Bayes
  val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial");

  // Score Model
  val train_predictionAndLabel = training.map(p => (model.predict(p.features), p.label));
  val test_predictionAndLabel = test.map(p => (model.predict(p.features), p.label));
  val train_accuracy = 1.0 * train_predictionAndLabel.filter(x => x._1 == x._2).count() / training.count();
  val test_accuracy = 1.0 * test_predictionAndLabel.filter(x => x._1 == x._2).count() / test.count();

  // Print Accuracies
  println("Accuracy of %s:", name);
  println(train_accuracy);
  println(test_accuracy);
}

for(l <- 1 to 3) {
    train_and_score_model(data_a, "set A");
    train_and_score_model(data_b, "set B");
    train_and_score_model(data_c, "set C");
    train_and_score_model(data_d, "set D");
    train_and_score_model(data_e, "set E");
    train_and_score_model(data_f, "set F");
}
