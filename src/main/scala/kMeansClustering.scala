import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.Numeric.Implicits._
import scala.math.{pow, sqrt}


object kMeansClustering {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val conf = new SparkConf().setAppName("test").setMaster("local[4]")
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    /* Run k means on Income Dataset and determine how good the the results are using class distribution,
    * accuracy, and other metrics such as true positive, false positive, etc.*/
    val dataset = readIncomeData
    val respData = readRespData
    val clusteredDataRaw = kmeans(dataset, 2, .01)
    val clusteredData = joinIncomeResp(clusteredDataRaw, respData)
    val classDist = getClassDistributions(clusteredData)
    val accuracy = getAccuracy(clusteredData)
    val moreMetrics = getMoreMetrics(clusteredData)

    // Since this is a clustering task, the "positive" i.e. "1" class will switch.
    // However, in all runs of our clustering we found that we cluster WAY more points into one group than the other.
    println(s"Class Distribution: $classDist")
    println(s"Accuracy: ${accuracy * 100}%")
    println(s"TP, FP, TN, FN: $moreMetrics")

    runTests()

  }

  /**
   * Minimal test suite for functions used in implementing k means algorithm
   *
   * @return
   */
  def runTests = () => {
    // Setup
    val dataset =
      sc.parallelize(List(
        ((0, List(1.0, 2.0)), 1),
        ((1, List(3.0, 4.0)), 1),
        ((2, List(5.0, 6.0)), 2),
        ((3, List(7.0, 8.0)), 1),
        ((4, List(9.0, 10.0)), 2),
        ((5, List(11.0, 12.0)), 1),
        ((6, List(13.0, 14.0)), 2)
      ))
    val dataWithoutCluster = sc.parallelize(List(
      List(1.0, 2.0),
      List(3.0, 4.0),
      List(5.0, 6.0)
    ))
    val dataWithId = sc.parallelize(List(
      (1, List(1.0, 2.0)),
      (2, List(4.0, 8.0)),
      (3, List(3.0, 6.0))
    ))
    val actualCentroids = List(List(5.5, 6.5), List(9.0, 10.0))
    val offCentroids = List(List(6, 6.5), List(9.0, 10.0))

    // getCentroids
    assert(actualCentroids == getCentroids(dataset))

    // shouldStop (if this passes, getSumOfSquares must be correct)
    assert(shouldStop(getCentroids(dataset), actualCentroids, 0))
    assert(shouldStop(getCentroids(dataset), offCentroids, 0.25))
    assert(!shouldStop(getCentroids(dataset), offCentroids, 0.20))

    // randomCentroids
    val randomCentroids = getRandomCentroids(dataWithoutCluster, 2)
    assert(randomCentroids.length == 2)
    assert(randomCentroids.head.head >= 1.0)
    assert(randomCentroids.head.head <= 5.0)
    assert(randomCentroids(1).head >= 1.0)
    assert(randomCentroids(1).head <= 5.0)
    assert(randomCentroids.head(1) >= 2.0)
    assert(randomCentroids.head(1) <= 6.0)
    assert(randomCentroids(1)(1) >= 2.0)
    assert(randomCentroids(1)(1) <= 6.0)

    // standardizeData
    val standardizedDataSet = standardizeData(dataWithId).collect().toList
    standardizedDataSet.map(_._2)
      .reduce((a, b) => (a, b).zipped.map(_ + _))
      .map(_ / 3)
      .foreach(x => {
        assert(x < 0.05)
      })
    // Assuming mean from above ~ 0, no need to calculate it...
    standardizedDataSet.map(_._2)
      .reduce((a, b) => (a, b).zipped.map(pow(_, 2) + pow(_, 2)))
      .map(x => sqrt(x / 3))
      .foreach(x => {
        assert(x < 2)
      })
  }

  /**
   * Read main (Income) dataset to run our implementation on,
   * preserving only numeric columns and adding an identifier to each observation
   *
   * @return
   */
  def readIncomeData: RDD[(Int, List[Double])] = {
    val income = sc.textFile("income.csv")
      .map(x => {
        List(
          x.split(",")(0)
            .trim()
            .toDouble,
          x.split(",")(3)
            .trim()
            .toDouble,
          x.split(",")(9)
            .trim()
            .toDouble,
          x.split(",")(10)
            .trim()
            .toDouble,
          x.split(",")(11)
            .trim()
            .toDouble,
          x
            .split(",")(0)
            .trim().toDouble)
      }).
      zipWithIndex().map({ case (lst, lng) => (lng.toInt, lst) })
    income
  }

  /**
   * Read Income dataset for ground truth for testing our results against
   *
   * @return
   */
  def readRespData: RDD[(Int, Int)] = {
    val income = sc.textFile("income.csv")
      .map(x => {
        x.split(",")(13).trim().toInt
      })
      .zipWithIndex()
      .map({ case (int, lng) => (lng.toInt, int) })
    income
  }

  /**
   * Join our results with the ground truth
   *
   * @param clustRes Our results
   * @param respData Ground truth
   * @return
   */
  def joinIncomeResp(clustRes: RDD[(Int, Int)], respData: RDD[(Int, Int)]): RDD[(Int, Int)] = {
    clustRes.join(respData).map({ case (x, y) => (y._2, y._1) })
  }

  /**
   * Helper function to get class distribution
   *
   * @param kmeansRes Result from running our k means algorithm on a dataset joined with ground truth dataset
   * @return
   */
  def getClassDistributions(kmeansRes: RDD[(Int, Int)]): List[Int] = {
    kmeansRes.map(x => (x._2, x._1)).groupByKey().map(x => x._2.toList.size).collect().toList
  }

  /**
   * Helper function to get accuracy score
   *
   * @param kmeansRes Result from running our k means algorithm on a dataset joined with ground truth dataset
   * @return
   */
  def getAccuracy(kmeansRes: RDD[(Int, Int)]): Double = {
    val totSize = kmeansRes.collect.length
    val filtered = kmeansRes.filter({ case (real, pred) => real.toInt == pred }).collect.length
    1.0 * filtered / totSize
  }

  /**
   * Helper function to get true positives, false positives, true negatives, false negatives
   *
   * @param kmeansRes Result from running our k means algorithm on a dataset joined with ground truth dataset
   * @return
   */
  def getMoreMetrics(kmeansRes: RDD[(Int, Int)]): (Int, Int, Int, Int) = {
    val TP = kmeansRes.filter({ case (real, pred) => real.toInt == pred & real.toInt == 1 }).collect.length
    val FP = kmeansRes.filter({ case (real, pred) => real.toInt != pred & pred == 1 }).collect.length
    val TN = kmeansRes.filter({ case (real, pred) => real.toInt == pred & real.toInt == 0 }).collect.length
    val FN = kmeansRes.filter({ case (real, pred) => real.toInt != pred & pred == 0 }).collect.length
    (TP, FP, FN, TN)
  }


  /**
   *
   * K-Means is an algorithm that takes in a dataset and a constant
   * k and returns k clusters (partitions of the data).
   *
   * @param dataSet
   * @param k
   */
  def kmeans(dataSet: RDD[(Int, List[Double])], k: Int, epsilon: Double): RDD[(Int, Int)] = {
    // Initialize centroids randomly using the dataset
    var oldCentroids = getRandomCentroids(dataSet.map(_._2), k)
    var iterations = 0

    // Modify dataset to store closest cluster for each observation
    var dataWithCluster: RDD[((Int, List[Double]), Int)] =
      dataSet.map(x => (x, findCluster(x._2, oldCentroids)))
    var centroids = getCentroids(dataWithCluster)

    // Reclassify until there are no changes or max iterations is exceeded
    while (!shouldStop(oldCentroids, centroids, epsilon)) {
      iterations += 1
      dataWithCluster = dataWithCluster.map(x => (x._1, findCluster(x._1._2, centroids)))
      oldCentroids = centroids
      centroids = getCentroids(dataWithCluster)
    }
    dataWithCluster.map(x => (x._1._1, x._2))
  }

  /**
   * For any given observation in a dataset, find a centroid closest to it using squared difference
   *
   * @param obs       : observation to find cluster for
   * @param centroids : existing list of centroids to assign cluster from
   * @return cluster id
   */
  def findCluster(obs: List[Double], centroids: List[List[Double]]): Int = {
    var minDiff = Double.MaxValue
    var cluster = 0
    centroids.indices.foreach(i => {
      val squared_diff = getSumOfSquares(centroids(i), obs)
      if (squared_diff < minDiff) {
        minDiff = squared_diff
        cluster = i
      }
    })
    cluster
  }

  /**
   * Helper function to get the sum of square differences between corresponding elements of two lists
   *
   * @param first
   * @param second
   * @return
   */
  def getSumOfSquares(first: List[Double], second: List[Double]): Double = {
    var squaredDiff = 0.0
    first.indices.foreach(i => {
      squaredDiff += pow(first(i) - second(i), 2)
    })
    squaredDiff
  }

  /**
   * Returns True or False if k-means is done. K-means terminates when the centroids
   * stop changing (or there is a negligible difference, using epsilon).
   *
   * @param oldCentroids :
   * @param centroids    :
   * @param iterations   :
   */

  def shouldStop(oldCentroids: List[List[Double]], centroids: List[List[Double]], epsilon: Double): Boolean = {
    var totSS = 0.0
    oldCentroids.indices.foreach(j => {
      totSS += getSumOfSquares(centroids(j), oldCentroids(j))
    })
    totSS <= epsilon
  }

  /**
   * Determine new centroids using the observation in dataset and their associated cluster id
   *
   * @param dataSet : Dataset to use to determine centroids
   * @return List of new centroids
   */
  def getCentroids(dataSet: RDD[((Int, List[Double]), Int)]): List[List[Double]] = {
    val centroids = dataSet.map(x => (x._2, x._1._2))
      .mapValues(value => (value, 1))
      .reduceByKey {
        case ((sumL, countL), (sumR, countR)) =>
          ((sumL, sumR).zipped.map(_ + _), countL + countR)
      }
      .values
      .map(x => x._1.map(y => y / x._2))
      .collect()
      .toList
    centroids
  }

  /**
   * Generates k number of centroids with each centroid containing random
   * double values corresponding to the values in any given observation
   *
   * For example: In dataset A = [(x1, y1, z1)...(x_i, y_i, z_i)] and k = 2, the following
   * must be true:
   * return [(a, b, c), (e, f, g)] where:
   * min(col1) ≤ a, e ≤ max(col1)
   * min(col2) ≤ b, f ≤ max(col2)
   * min(col3) ≤ c, g ≤ max(col3)
   *
   * @param dataSet : dataset for which random centroids need to be generated
   * @param k       : Number of centroid to generate
   * @return Returns a list of size k of random centroids
   */

  def getRandomCentroids(dataSet: RDD[List[Double]], k: Int): List[List[Double]] = {
    var inc = 0
    val numVariables = dataSet.take(1)(0).size
    val r = scala.util.Random
    var resultList = List.fill(k)(List.fill(numVariables)(0.0))

    while (inc < numVariables) {
      // just initialize to first value in column
      var max: Double = dataSet.take(1)(0)(inc)
      var min = max

      // sort by current column decreasing take top for max
      max = dataSet.sortBy(x => x(inc) * -1).take(1)(0)(inc)

      // sort by current column increasing take top for min
      min = dataSet.sortBy(x => x(inc)).take(1)(0)(inc)

      var i = 0
      while (i < k) {
        // r.nextDouble is 0.0 - 1.0. Multiply this by the range and add to the min
        var result = (r.nextDouble() * (max - min)) + min
        // update our resultList
        resultList = resultList.updated(i, resultList(i).updated(inc, result))
        i += 1
      }
      inc += 1
    }
    // return our result list
    resultList
  }

  /**
   * Function to standardize a given dataset
   *
   * @param dataSet dataset to standardize
   * @return
   */
  def standardizeData(dataSet: RDD[(Int, List[Double])]): RDD[(Int, List[Double])] = {

    val datasetSumAndCount = dataSet.map(x => (x._2, 1)).reduce {
      case ((sumL, countL), (sumR, countR)) =>
        ((sumL, sumR).zipped.map(_ + _), countL + countR)
    }
    val dataSetMean = datasetSumAndCount._1.map(_ / datasetSumAndCount._2)

    val datasetVariance = dataSet.map(_._2).reduce {
      case (sumL, sumR) =>
        val _varL = (sumL, dataSetMean).zipped.map(_ - _).map(pow(_, 2))
        val _varR = (sumR, dataSetMean).zipped.map(_ - _).map(pow(_, 2))
        (_varL, _varR).zipped.map(_ + _)
    }

    val dataSetSd = datasetVariance.map(x => sqrt(x / (datasetSumAndCount._2 - 1)))
    dataSet.map { case (id, obs) =>
      (id, ((obs, dataSetMean).zipped.map(_ - _), dataSetSd).zipped.map(_ / _))
    }
  }

}
