package org.apache.spark.ml.customlinreg

import com.google.common.io.Files
import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers.should


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark{
  val params = DenseVector[Double](1.5, 0.3, -0.7)
  val dataSize = 50
  val delta = 0.01
  val epsMSE = 0.0001

  lazy val X = DenseMatrix.rand[Double](dataSize, params.size)
  lazy val y = X * params + 0.001 * DenseVector.rand[Double](dataSize)
  lazy val data = dataGeneration(X, y)

  private def dataGeneration(X: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
    import sqlc.implicits._
    lazy val df: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    lazy val df_ = df(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq.toDF("x_1", "x_2", "x_3", "y")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x_1", "x_2", "x_3")).setOutputCol("x")
    lazy val data = assembler.transform(df_).select("x", "y")
    data
  }

  private def validateModel(model: LinearRegressionModel): Unit = {
    val dfWithPred = model.transform(data)
    val regr = new RegressionEvaluator()
      .setLabelCol("y")
      .setPredictionCol("pred")
      .setMetricName("mse")
    val mse = regr.evaluate(dfWithPred)
    mse should be <= epsMSE
  }

  "Estimator" should s"have params: $params" in {
    val linReg = new LinearRegression()
      .setInputCol("x")
      .setOutputCol("y")
    val model = linReg.fit(data)
    val w = model.getW
    w(0) should be(params(0) +- delta)
    w(1) should be(params(1) +- delta)
    w(2) should be(params(2) +- delta)
  }

  "Model's" should "MSE is less epsMSE" in {
    val linReg = new LinearRegression()
      .setInputCol("x")
      .setOutputCol("y")
      .setPredCol("pred")
    val model = linReg.fit(data)
    validateModel(model)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
     Array(
        new LinearRegression()
           .setInputCol("x")
           .setOutputCol("y")
           .setPredCol("pred")
      )
    )
    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
                         .fit(data)
                         .stages(0)
                         .asInstanceOf[LinearRegressionModel]
    validateModel(reRead)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinearRegression()
          .setInputCol("x")
          .setOutputCol("y")
          .setPredCol("pred")
      )
    )
    val model = pipeline.fit(data)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
    validateModel(reRead.stages(0).asInstanceOf[LinearRegressionModel])
  }
}
