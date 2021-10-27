package linear_regression
import breeze.linalg._
import breeze.numerics.pow
import breeze.stats.mean

object MeanSquaredError {
  def calculate(yTrue: DenseVector[Double], yPred: DenseVector[Double]): Double = {
    mean(pow(yTrue - yPred, 2))
  }
}

class LinearRegression() {
  var weight: DenseVector[Double] = DenseVector[Double]()

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val X_ = DenseMatrix.horzcat(X, DenseMatrix.ones[Double](X.rows, 1))
    X_ * weight
  }

  def fit(X: DenseMatrix[Double], y: DenseVector[Double]): Unit = {
    val X_ = DenseMatrix.horzcat(X, DenseMatrix.ones[Double](X.rows, 1))
    weight = pinv(X_.t * X_) * X_.t * y
  }
}

object TrainTestSplit {
  def split(data: DenseMatrix[Double], trainSize:Double = 0.75): (DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double], DenseVector[Double]) = {
    val numTrainRows = (data.rows * trainSize).toInt
    val trainX = data(0 until numTrainRows, 0 until data.cols - 1)
    val trainY = data(0 until numTrainRows, -1)
    val validX = data(numTrainRows until data.rows, 0 until data.cols - 1)
    val validY = data(numTrainRows until data.rows, -1)
    (trainX, trainY, validX, validY)
  }
}


