/**
 *  Hello Open Source Community
 *  
 * 
 * ***/






import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.{ SparkContext, SparkConf }  
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import scala.collection.immutable.HashMap
import org.apache.spark.mllib.regression.LabeledPoint
import scala.math.pow
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics


object SparApp {
  
def main(args: Array[String]): Unit = {
  

  //val conf = new SparkConf().setAppName("SparkMe Application")
val conf = new SparkConf().setAppName("SOME APP NAME").setMaster("local[2]").set("spark.executor.memory","1g")
val sc = new SparkContext(conf)
val categories = Array("alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med")
val categoryMap = categories.zipWithIndex.toMap  
val numFeatures = pow(2,18).toInt  // default is 2**20, so reduce on smaller machines 

def tokenize(line: String): Array[String] = {
    line.split("""\W+""").map(_.toLowerCase)
}

def prepareData(typ: String) = {
 
    categories.map(category => {
        val wordsData = sc.wholeTextFiles("/user/zeppelin/20newsgroups/20news-bydate-" + typ + "/" + category)
                                          
                          .map(message => tokenize(message._2).toSeq)

        val hashingTF = new HashingTF(numFeatures)
        val featuredData = hashingTF.transform(wordsData).cache()

        val idf = new IDF().fit(featuredData)
        val tfidf = idf.transform(featuredData)
        tfidf.map(row => LabeledPoint(categoryMap(category),row))
    }).reduce(_ union _)
}

val twenty_train = prepareData("train").cache()
val twenty_test  = prepareData("test").cache()

/***
 * 
 * Create Model
 * 
 */

val model = new LogisticRegressionWithLBFGS().setNumClasses(4).run(twenty_train)
/**
 * 
 * Validate the model
 */

val toInt = {i:Double => i.asInstanceOf[Number].intValue}
def printMetrics(metrics: MulticlassMetrics, categories: Array[String]) = {
    println("")
    println("CONFUSION MATRIX")
    println(metrics.confusionMatrix)
    println("")
    println("CATEGORY                 PRECISION  RECALL")
    
    metrics.labels.map(toInt).foreach { i => 
        val l = categories(i)
        val p = metrics.precision(i)
        val r = metrics.recall(i)
        println(f"$l%22s:  $p%2.3f      $r%2.3f")
    }
    println("")
}

val predictionsAndLabels = twenty_test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

val metrics = new MulticlassMetrics(predictionsAndLabels)

printMetrics(metrics, categories)

}

}






   
