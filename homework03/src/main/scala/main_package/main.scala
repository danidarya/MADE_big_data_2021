package main_package
import breeze.linalg.{csvread, csvwrite}
import linear_regression.{LinearRegression, MeanSquaredError, TrainTestSplit}
import java.io.{BufferedWriter, File, FileWriter}

object Main{
  def main(args: Array[String]): Unit = {

    val logFile = new File("./src/log.txt")
    val bw = new BufferedWriter(new FileWriter(logFile))

    if (args.length < 2) {
      bw.write("There must be 2 paths to files: to train data and to test data.\n")
      bw.close()
      return
    }

    val trainData = csvread(new File(args(0)), separator = ',', skipLines = 1)
    val trainSize = 0.75
    val (trainX, trainY, validX, validY) = TrainTestSplit.split(trainData,trainSize) //(trainData(::, 0 until trainData.cols - 1), trainData(::, -1))
    bw.write(f"Dataset splitted for train and valid parts. Train size is ${(trainData.rows * trainSize).toInt} rows, " +
      f"validation size is ${trainData.rows - (trainData.rows * trainSize).toInt} rows.\n")
    val testData = csvread(new File(args(1)), separator = ',', skipLines = 1)
    val (testX, testY) = (testData(::, 0 until trainData.cols - 1), testData(::, -1))

    val lr = new LinearRegression()
    lr.fit(trainX, trainY)
    bw.write("Model is fitted.\n")
    val predY = lr.predict(validX)
    bw.write(f"Mean Squared Error on validation is ${MeanSquaredError.calculate(validY, predY)}.\n")
    val predTestX = lr.predict(testX)
    bw.write("Predictions were done and written to /src/predictions.csv.\n")
    bw.write(f"Mean Squared Error on prediction is ${MeanSquaredError.calculate(testY, predTestX)}.\n")
    bw.close()
    csvwrite(new File("./src/predictions.csv"), predTestX.toDenseMatrix.t)
  }
}