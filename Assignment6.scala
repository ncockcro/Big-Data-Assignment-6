import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._ // optional
import org.apache.spark.mllib.rdd._

val file = sc.textFile("/home/nick/input/animals.csv")
val header = file.first()
val rawData = file.filter(x=> x!=header )

val Data = rawData.map{x=>
		val values = x.split(',').map(x=> x.toDouble) // splitting on , and converting everything
		val featureVector = Vectors.dense(values.init) // With values, we create a featureVector with the first 5 valjues
		val label = values.last	// Get the last column from the values
		println(featureVector)
		println(label)
		LabeledPoint(label, featureVector)
		}

val categoricalFeatureInfo = Map[Int, Int]((3,4)) // First field is the index from the file and the second is how many attributes
val model = DecisionTree.trainClassifier(Data, 2, categoricalFeatureInfo, "gini", 5, 100)
val testAnimal = Vectors.dense(0,20,4,0) // Testing an animal
val predicition  = model.predict(testAnimal)

println(" Model Tree:\n " + model.toDebugString)
