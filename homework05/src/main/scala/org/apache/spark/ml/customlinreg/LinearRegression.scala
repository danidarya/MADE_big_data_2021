package org.apache.spark.ml.customlinreg

import breeze.linalg.{functions, sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader,
  DefaultParamsWritable, DefaultParamsWriter, Identifiable,
  MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils, SchemaUtils}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.{DoubleType, StructType}

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val predCol: Param[String] = new Param[String](
    this, "pred", "prediction column")
  def setPredCol(value: String): this.type = set(predCol, value)
  def getPredCol: String = $(predCol)
  setDefault(predCol, "pred")

  val eps = new DoubleParam(this, "eps", "eps for error")
  def setEps(value: Double) : this.type = set(eps, value)
  setDefault(eps -> 0.0001)

  val lr = new DoubleParam(this, "lr", "learning rate")
  def setLearningRate(value: Double) : this.type = set(lr, value)
  setDefault(lr -> 0.1)


  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)
    schema
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("linearRegression"))
  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
    val dataVectorOnes = dataset.withColumn("ones", lit(1))
    val assembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), "ones", $(outputCol)))
      .setOutputCol("dataset_and_vector_ones")

    val dataTrain: Dataset[Vector] = assembler.transform(dataVectorOnes).select(col="dataset_and_vector_ones").as[Vector]
    val nCols = MetadataUtils.getNumFeatures(dataset, $(inputCol)) + 1
    var prevW = new BreezeDenseVector(data=Array.fill[Double](nCols)(Double.PositiveInfinity))
    var w = BreezeDenseVector.rand[Double](nCols)

    while (functions.euclideanDistance(w.toDenseVector,prevW.toDenseVector) > $(eps)) {
      val summary = dataTrain.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until w.size).toDenseVector
          val dw = 2.0 * (X * (sum(X * w) - v.asBreeze(-1)))
          summarizer.add(mllib.linalg.Vectors.fromBreeze(dw))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)
      prevW = w.copy
      w = w - summary.mean.asBreeze * $(lr)
    }
    copyValues(new LinearRegressionModel(w).setParent(this))
  }
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel (override val uid: String,
                               val w: BreezeDenseVector[Double])
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable{

  def getW: BreezeDenseVector[Double] = {
    w
  }
  def this(w: BreezeDenseVector[Double]) =
    this(Identifiable.randomUID("linearRegressionModel"), w)
  override def copy(extra: ParamMap): LinearRegressionModel = defaultCopy(extra)
  override def transform(dataset: Dataset[_]): DataFrame = {
    val vectW = Vectors.fromBreeze(w(0 to w.size - 2))
    val transformUdf = {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x : Vector) => { x.dot(vectW) + w(-1) }
      )
    }
    dataset.withColumn($(predCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
      override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      sqlContext.createDataFrame(Seq(Tuple1(Vectors.fromBreeze(w))))
        .write.parquet(path + "/data")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/data")
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      val w : BreezeDenseVector[Double] = vectors.select(vectors("_1")
        .as[Vector]).first().asBreeze.toDenseVector
      val model = new LinearRegressionModel(w)
      metadata.getAndSetParams(model)
      model
    }
  }
}