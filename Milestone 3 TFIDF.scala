// Databricks notebook source
//import all the related spark libraries
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StopWordsRemover

// val data_dayinlife = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/yf4h591t1494968726246/tfidf_day.csv");
val data_dayinlife = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("/FileStore/tables/636pdoan1495000764043/df_day__1_-86b3b.csv");

// COMMAND ----------

/*
  TOKENIZE STRING INTO WORDS
*/
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions._

// Tokenize the DayInLife string
// val tokenizer = new RegexTokenizer().setInputCol("DayInLife").setOutputCol("words").setPattern("\\W");
val tokenizer = new RegexTokenizer().setInputCol("DayInLife").setOutputCol("words").setPattern("[^a-zA-Z]"); // Only keep a-z and A-Z. Don't keep numbers
val countTokens = udf { (words: Seq[String]) => words.length };
val tokenized = tokenizer.transform(data_dayinlife);
tokenized.select("DayInLife", "words")
  .withColumn("tokens", countTokens(col("words"))).show(10);

// COMMAND ----------

/*
  TEXT NORMALIZATION: STOP WORD REMOVAL
*/
import org.apache.spark.ml.feature.StopWordsRemover

val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered");
val filtered_day_in_life = remover.transform(tokenized).select("dog_SubStatusCode", "filtered");
filtered_day_in_life.show(20);

// COMMAND ----------

/*
  TEXT NORMALIZATION: STEMMING
*/
// NEED TO DO THIS STILL

// COMMAND ----------

/*
  FEATURE EXTRACTION: TF-IDF
*/
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("rawFeatures").setNumFeatures(10)

val featurizedData = hashingTF.transform(filtered_day_in_life);
// featurizedData.show(10);
featurizedData.printSchema()
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("idf_features")
val idfModel = idf.fit(featurizedData)


// COMMAND ----------

val rescaledData = idfModel.transform(featurizedData)
// rescaledData.select("dog_SubStatusCode", "features").show()
val newData = rescaledData.select("dog_SubStatusCode", "idf_features");
newData.show();

// COMMAND ----------

// def convert_data(data: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]) : org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = {

//   // Assembles Vector with all features except the label column
//   val assembler = new VectorAssembler()
//     .setInputCols(Array("StealsFood-0", "StealsFood-1", "StealsFood-2", "StealsFood-3", "StealsFood-4", "StealsFood-5", "EnergyLevel-0", "EnergyLevel-1", "EnergyLevel-2", "EnergyLevel-3", "EnergyLevel-4", "EnergyLevel-5", "JumpOnPeople-0", "JumpOnPeople-1", "JumpOnPeople-2", "JumpOnPeople-3", "JumpOnPeople-4", "JumpOnPeople-5", "AttendsClasses-0", "AttendsClasses-1", "AttendsClasses-2", "AttendsClasses-3", "AttendsClasses-4", "AttendsClasses-5", "FriendlyWAnimals-0", "FriendlyWAnimals-1", "FriendlyWAnimals-2", "FriendlyWAnimals-3", "FriendlyWAnimals-4", "FriendlyWAnimals-5", "Health-0", "Health-1", "Health-2", "Health-3", "Health-4", "Health-5", "AttendsHomeSwitches-0", "AttendsHomeSwitches-1", "AttendsHomeSwitches-2", "AttendsHomeSwitches-3", "AttendsHomeSwitches-4", "AttendsHomeSwitches-5", "TrafficFear-0", "TrafficFear-1", "TrafficFear-2", "TrafficFear-3", "TrafficFear-4", "TrafficFear-5", "Sex", "StaysOnCommand-0", "StaysOnCommand-1", "StaysOnCommand-2", "StaysOnCommand-3", "StaysOnCommand-4", "StaysOnCommand-5", "RaidsGarbage-0", "RaidsGarbage-1", "RaidsGarbage-2", "RaidsGarbage-3", "RaidsGarbage-4", "RaidsGarbage-5", "BehavesWellClass-0", "BehavesWellClass-1", "BehavesWellClass-2", "BehavesWellClass-3", "BehavesWellClass-4", "BehavesWellClass-5", "Stairs-0", "Stairs-1", "Stairs-2", "Stairs-3", "Stairs-4", "Stairs-5", "Breed-belgian shepherd", "Breed-german shepherd", "Breed-golden labrador", "Breed-golden retriever", "Breed-labrador retriever", "Color-Black", "Color-Black/tan", "Color-Golden", "Color-Sable", "Color-Yellow", "ExerciseAmount-20", "ExerciseAmount-40", "ExerciseAmount-60", "ExerciseAmount-90", "ExerciseAmount-120", "ExerciseAmount-180", "ExerciseAmount-440", "Housemanners-0", "Housemanners-1", "Housemanners-2", "Housemanners-3", "Housemanners-4", "Housemanners-5"))
//     .setOutputCol("features");

//   // Select only necessary columns
//   val output = assembler.transform(data).select("features", "dog_SubStatusCode");

//   // Rename column outputs for classifier..
//   val newNames = Seq("features", "label");
//   val output2 = output.toDF(newNames: _*);
//   output2
// }

val newNames = Seq("label", "features");
val newData_2 = newData.toDF(newNames: _*);
newData_2.show();

// COMMAND ----------

// Train Function
def train_and_score_model(data: org.apache.spark.sql.DataFrame, name: String, lambda: Double, model_type: String, split_random: Boolean) : Unit = {

  data.show()

  // Split Data
  var Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 0);
  // if(split_random == true) { // If splitMode true, do randomly
  //   splits = data.randomSplit(Array(0.8, 0.2));
  // }

//   test.show();

//   training.show();

//   Train Naive Bayes
  val model = new NaiveBayes().fit(training);

//   // Score Model
//   // Select example rows to display.
//   val predictions = model.transform(test);

//   val evaluator = new MulticlassClassificationEvaluator()
//     .setLabelCol("label")
//     .setPredictionCol("prediction")
//     .setMetricName("accuracy")

//   val accuracy = evaluator.evaluate(predictions)
//   println("Test set accuracy = " + accuracy)
}


// COMMAND ----------

// train_and_score_model(newData_2, "set A", 1.0, "multinomial", false);
