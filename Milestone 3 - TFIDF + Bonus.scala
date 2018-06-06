// Databricks notebook source
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, StopWordsRemover}
import org.apache.spark.sql.Row


// COMMAND ----------

val data_dayinlife  = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
//     .load("/FileStore/tables/sf9tsexq1495155683555/puppy_and_dayinlife.csv")
    .load("/FileStore/tables/inelloel1495157937942/bonus_b3-c1fc6.csv")

data_dayinlife.printSchema()

// COMMAND ----------

// Create data to labels and features column
def convert_data(data: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row], feature_arr: Array[String]) : org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = {
  
  // Assembles Vector with all features except the label column
  val assembler = new VectorAssembler().setInputCols(feature_arr).setOutputCol("features");
  
  // Select only necessary columns
  val output = assembler.transform(data).select("features", "dog_SubStatusCode_x").toDF("features", "label");
  output
}

// COMMAND ----------

// Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
def apply_tfidf_on_col(data: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row], col_input_name: String, col_output_name: String) : org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = {
  val tokenizer = new Tokenizer().setInputCol(col_input_name).setOutputCol(col_input_name + "words")
  val remover = new StopWordsRemover().setInputCol(col_input_name + "words").setOutputCol(col_input_name + "filtered")
  val hashingTF = new HashingTF().setNumFeatures(100).setInputCol(tokenizer.getOutputCol).setOutputCol(col_input_name + "features_tf")
  val idf = new IDF().setInputCol(col_input_name + "features_tf").setOutputCol(col_output_name)
  val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf))

  // Fit the pipeline to training documents.
  val transform_data = pipeline.fit(data)
  val data_transformed = transform_data.transform(data);
  data_transformed
}

// COMMAND ----------

val data_transformed =
  apply_tfidf_on_col(
    apply_tfidf_on_col(
      apply_tfidf_on_col(
        data_dayinlife, "DayInLife", "features_dil"
      ),
      "ExerciseType", "features_et"
    ),
    "FoodType", "features_ft"
  );

val all_features_part3 = Array("features_dil");
val data_converted_part3 = convert_data(data_transformed, all_features_part3);

val all_features_b1 = Array("Color-Black", "Color-Black/tan", "Color-Golden", "Color-Sable", "Color-Yellow", "Health-0", "Health-1", "Health-2", "Health-3", "Health-4", "Health-5", "EnergyLevel-0", "EnergyLevel-1", "EnergyLevel-2", "EnergyLevel-3", "EnergyLevel-4", "EnergyLevel-5", "StealsFood-0", "StealsFood-1", "StealsFood-2", "StealsFood-3", "StealsFood-4", "StealsFood-5", "BehavesWellClass-0", "BehavesWellClass-1", "BehavesWellClass-2", "BehavesWellClass-3", "BehavesWellClass-4", "BehavesWellClass-5", "RaidsGarbage-0", "RaidsGarbage-1", "RaidsGarbage-2", "RaidsGarbage-3", "RaidsGarbage-4", "RaidsGarbage-5", "Stairs-0", "Stairs-1", "Stairs-2", "Stairs-3", "Stairs-4", "Stairs-5", "features_dil");
val data_converted_b1 = convert_data(data_transformed, all_features_b1);


val all_features_b3 = Array("features_et", "features_ft");
val data_converted_b3 = convert_data(data_transformed, all_features_b3);

val all_features_b4 = Array("Color-Black", "Color-Black/tan", "Color-Golden", "Color-Sable", "Color-Yellow", "Health-0", "Health-1", "Health-2", "Health-3", "Health-4", "Health-5", "EnergyLevel-0", "EnergyLevel-1", "EnergyLevel-2", "EnergyLevel-3", "EnergyLevel-4", "EnergyLevel-5", "StealsFood-0", "StealsFood-1", "StealsFood-2", "StealsFood-3", "StealsFood-4", "StealsFood-5", "BehavesWellClass-0", "BehavesWellClass-1", "BehavesWellClass-2", "BehavesWellClass-3", "BehavesWellClass-4", "BehavesWellClass-5", "RaidsGarbage-0", "RaidsGarbage-1", "RaidsGarbage-2", "RaidsGarbage-3", "RaidsGarbage-4", "RaidsGarbage-5", "Stairs-0", "Stairs-1", "Stairs-2", "Stairs-3", "Stairs-4", "Stairs-5", "features_et", "features_ft");
val data_converted_b4 = convert_data(data_transformed, all_features_b4);

val all_features_b5 = Array("Color-Black", "Color-Black/tan", "Color-Golden", "Color-Sable", "Color-Yellow", "Health-0", "Health-1", "Health-2", "Health-3", "Health-4", "Health-5", "EnergyLevel-0", "EnergyLevel-1", "EnergyLevel-2", "EnergyLevel-3", "EnergyLevel-4", "EnergyLevel-5", "StealsFood-0", "StealsFood-1", "StealsFood-2", "StealsFood-3", "StealsFood-4", "StealsFood-5", "BehavesWellClass-0", "BehavesWellClass-1", "BehavesWellClass-2", "BehavesWellClass-3", "BehavesWellClass-4", "BehavesWellClass-5", "RaidsGarbage-0", "RaidsGarbage-1", "RaidsGarbage-2", "RaidsGarbage-3", "RaidsGarbage-4", "RaidsGarbage-5", "Stairs-0", "Stairs-1", "Stairs-2", "Stairs-3", "Stairs-4", "Stairs-5", "features_dil", "features_et", "features_ft");
val data_converted_b5 = convert_data(data_transformed, all_features_b4);

// COMMAND ----------

// Train and test model on dataset
def train_and_print_accruacy(data: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row], fixedSplit: Boolean) : Unit = {
  // Split into training and test data
  var splits = data.randomSplit(Array(0.8, 0.2));
  if(fixedSplit == true) {
    splits = data.randomSplit(Array(0.8, 0.2), seed = 100L);
  }
  val Array(training, test) = splits;

  // Train on Naive Bayes Classifier
  val model = new NaiveBayes()
  classifier.setModelType("multinomial");

  val trained_model = model.fit(training);
  
  // Predict on test and training data
  val predictions_train = trained_model.transform(training).select("prediction", "label");
  val predictions_test = trained_model.transform(test).select("prediction", "label");

  // Get only the correct predictions
  val correct_preds_train = predictions_train.collect().map {x =>
    val correct = x(0) == x(1);
    correct
  };
  val correct_preds_test  = predictions_test.collect().map {x =>
    val correct = x(0) == x(1);
    correct
  };

  // Filter and print
  val num_correct_train = correct_preds_train.filter(x => x == true).length;
  val size_train = training.rdd.count
  val accuracy_train = num_correct_train * 100.0 / size_train;
  val num_correct_test = correct_preds_test.filter(x => x == true).length;
  val size_test = test.rdd.count
  val accuracy_test = num_correct_test * 100.0 / size_test;

  println(s"TRAIN: $num_correct_train out of $size_train is $accuracy_train");
  println(s"TEST:  $num_correct_test out of $size_test is $accuracy_test");

}

// COMMAND ----------

println("PART 3");
train_and_print_accruacy(data_converted_part3, true);
train_and_print_accruacy(data_converted_part3, true);
train_and_print_accruacy(data_converted_part3, true);
train_and_print_accruacy(data_converted_part3, false);
train_and_print_accruacy(data_converted_part3, false);
train_and_print_accruacy(data_converted_part3, false);

// COMMAND ----------

println("PART B1");
train_and_print_accruacy(data_converted_b1, true);
train_and_print_accruacy(data_converted_b1, true);
train_and_print_accruacy(data_converted_b1, true);
train_and_print_accruacy(data_converted_b1, false);
train_and_print_accruacy(data_converted_b1, false);
train_and_print_accruacy(data_converted_b1, false);

println("PART B3");
train_and_print_accruacy(data_converted_b3, true);
train_and_print_accruacy(data_converted_b3, true);
train_and_print_accruacy(data_converted_b3, true);
train_and_print_accruacy(data_converted_b3, false);
train_and_print_accruacy(data_converted_b3, false);
train_and_print_accruacy(data_converted_b3, false);

println("PART B4");
train_and_print_accruacy(data_converted_b4, true);
train_and_print_accruacy(data_converted_b4, true);
train_and_print_accruacy(data_converted_b4, true);
train_and_print_accruacy(data_converted_b4, false);
train_and_print_accruacy(data_converted_b4, false);
train_and_print_accruacy(data_converted_b4, false);

println("PART B5");
train_and_print_accruacy(data_converted_b5, true);
train_and_print_accruacy(data_converted_b5, true);
train_and_print_accruacy(data_converted_b5, true);
train_and_print_accruacy(data_converted_b5, false);
train_and_print_accruacy(data_converted_b5, false);
train_and_print_accruacy(data_converted_b5, false);


// COMMAND ----------


