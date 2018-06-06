// Databricks notebook source
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)
val df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .load("/FileStore/tables/pmwjmwj21494482453756/df_day.csv")

val tokenizer = new Tokenizer().setInputCol("DayInLife").setOutputCol("words")
val regexTokenizer = new RegexTokenizer()
  .setInputCol("DayInLife")
  .setOutputCol("words")
  .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

val tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("words", "dog_SubStatusCode").take(3).foreach(println)
val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.select("words", "dog_SubStatusCode").take(3).foreach(println)

// COMMAND ----------

val f = sc.textFile("/FileStore/tables/xwzo00hu1494480938586/text.csv")

// COMMAND ----------

val wc = f.flatMap(l => l.split(" ")).map(word => (word,1)).reduceByKey(_ + _)

// COMMAND ----------

 //swap k,v to v,k to sort by word frequency
val wc_swap = wc.map(_.swap)

// COMMAND ----------

val hifreq_word250 = wc_swap.sortByKey(false,1)
hifreq_words.saveAsTextFile("hifreq_words20")
val top20 = hifreq_words20.take(20)
// convert array to RDD
val top20rdd = sc.parallelize(top20)
top20rdd.saveAsTextFile("hifreq_top20")

// COMMAND ----------


