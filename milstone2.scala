// Databricks notebook source
//import all the related spark libraries
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils

// COMMAND ----------

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

// Load in all Data
def load_file(filename: String) : org.apache.spark.rdd.RDD[String] = {
  val data = sc.textFile(filename);
  val header = data.first() // Extract Header
  data.filter(row => row != header) // Remove Header
}




// COMMAND ----------

val Data = load_file("/FileStore/tables/3h8jt8f81494478001964/df_day.csv")


// COMMAND ----------

// Parse Data into (Label, Features)
def parse_data(data: org.apache.spark.rdd.RDD[String]) : org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint] = {
  data.map { line =>
    val parts = line.split(',').map(_.toDouble)
    LabeledPoint(parts(0), Vectors.dense(parts.tail))
  }
}

// COMMAND ----------

val parsed = parse_data(Data);

// COMMAND ----------

// Parse Data into (Label, Features)
val parsedData = sentenceData.map { line =>
    val parts = line.split(',').map(_.toDouble)
    LabeledPoint(parts(0), Vectors.dense(parts.tail))
}


// COMMAND ----------

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

// Input data: Each row is a bag of words from a sentence or document.
val documentDF = sc.textFile("/FileStore/tables/xwzo00hu1494480938586/text.csv").map(Tuple1.apply).toDF("text")



// COMMAND ----------

// Learn a mapping from words to Vectors.
val word2Vec = new Word2Vec()
  .setInputCol("text")
  .setOutputCol("result")
  .setVectorSize(3)
  .setMinCount(0)

// COMMAND ----------

val model = word2Vec.fit(documentDF)


// COMMAND ----------

import org.apache.spark.mllib.classification.NaiveBayes

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.feature.{HashingTF, IDF, Normalizer}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

// COMMAND ----------

val conf = new SparkConf()
val sc = new SparkContext(conf)

// COMMAND ----------

import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.util.MLUtils

val data = MLUtils.loadLibSVMFile(sc, "/FileStore/tables/xwzo00hu1494480938586/text.csv")

val normalizer1 = new Normalizer()


// Each sample in data1 will be normalized using $L^2$ norm.
val data1 = data.map(x => (x.label, normalizer1.transform(x.features)))



// COMMAND ----------

import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)
val df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .load("/FileStore/tables/pmwjmwj21494482453756/df_day.csv")

// COMMAND ----------

val tokenizer = new Tokenizer().setInputCol("DayInLife").setOutputCol("words")

// COMMAND ----------

val wordsData = tokenizer.transform(df)

// COMMAND ----------

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
val hashingTF = new HashingTF()
  .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

// COMMAND ----------

val featurizedData = hashingTF.transform(wordsData)

// COMMAND ----------

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)

// COMMAND ----------

val rescaledData = idfModel.transform(featurizedData)
rescaledData.select("dog_SubStatusCode", "features").show()

// COMMAND ----------

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

// COMMAND ----------

val regexTokenizer = new RegexTokenizer()
  .setInputCol("DayInLife")
  .setOutputCol("words")
  .setPattern("\\W") 

// COMMAND ----------

val tokenized = tokenizer.transform(df)

// COMMAND ----------

tokenized.select("words", "dog_SubStatusCode").take(3).foreach(println)

// COMMAND ----------

val regexTokenized = regexTokenizer.transform(df)

// COMMAND ----------

regexTokenized.select("words", "dog_SubStatusCode").take(3).foreach(println)


// COMMAND ----------

import org.apache.spark.ml.feature.StopWordsRemover

val remover = new StopWordsRemover()
dd
remover.transform(regexTokenized.select("words", "dog_SubStatusCode").take(3).foreach(println)).show()

// COMMAND ----------

val hashingTF = new HashingTF().setInputCol( "words" ).setOutputCol( "rawFeatures" )
val featurizedData = hashingTF.transform( wordsData )

// COMMAND ----------

val idf = new IDF().setInputCol( "rawFeatures" ).setOutputCol( "features" )

// COMMAND ----------

val idfModel = idf.fit( featurizedData )

// COMMAND ----------

val sentenceDF = sc.textFile("/FileStore/tables/xwzo00hu1494480938586/text.csv")

// COMMAND ----------

val lines = sc.textFile(args(0)) // RDD CREATED

// COMMAND ----------


