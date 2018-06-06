// Databricks notebook source
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

val input = sc.textFile("/FileStore/tables/xwzo00hu1494480938586/text.csv").map(line => line.split(" ").toSeq)

val word2vec = new Word2Vec()

val model = word2vec.fit(input)

// COMMAND ----------

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

val input = sc.textFile("/FileStore/tables/xwzo00hu1494480938586/text.csv").map(line => line.split(" ").toSeq)

val word2Vec = new Word2Vec()
  .setVectorSize(3)
  .setMinCount(0)

val model = word2vec.fit(input.map(_._1)
)

// COMMAND ----------

val result = model.transform(input)
result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
  println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }

// COMMAND ----------

import org.apache.spark.ml.feature.StandardScaler

val dataFrame = spark.read.format("libsvm").load("/FileStore/tables/pmwjmwj21494482453756/df_day.csv")

val scaler = new StandardScaler()


// Compute summary statistics by fitting the StandardScaler.
val scalerModel = scaler.fit(dataFrame)


// COMMAND ----------


