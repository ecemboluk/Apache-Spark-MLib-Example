
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.mean;

public class Application {
    public static void main(String[] args) {
        int counter = 0;
        int avg = 0;
        SparkSession sparkSession = SparkSession.builder()
                .appName("Diabetes Analysis")
                .master("local") // Master machine name We write local because we use our computer as the master machine.
                .getOrCreate();

        Dataset<Row> dataset = sparkSession.read()
                .format("csv")
                .option("header", "true")
                .load("your file path/diabetes.csv");

        // Explorer Data Analysis

        // Dataset<Row> outcome = dataset.groupBy("Outcome").count();
        // long count = dataset.filter("Pregnancies == 0").count();
        // System.out.println(count);
        // dataset.select(mean("Age")).show();
        // dataset.select(mean("BMI")).show();
        //
        // Dataset<Row> groupedAge = dataset.groupBy("Outcome", "Age").count();
        // groupedAge.show();
        //
        // Dataset<Row> diabetes = dataset.filter("Outcome == 1");
        // diabetes.show();
        // diabetes.select(mean("Age")).show();
        // diabetes.select(mean("BMI")).show();

        String[] vowels = {"Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"};

        List<String> headers = Arrays.asList(vowels);
        for(String h:headers){
            if(h.equals("Outcome")){
                StringIndexer tmpIndex = new StringIndexer().setInputCol(h).setOutputCol("label");
                dataset = tmpIndex.fit(dataset).transform(dataset);
            }
            else {
                StringIndexer tmpIndex = new StringIndexer().setInputCol(h).setOutputCol(h.toLowerCase() + "_cat");
                dataset = tmpIndex.fit(dataset).transform(dataset);
            }
        }


        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(toLowerList(headers))
                .setOutputCol("features");
        Dataset<Row> transform = vectorAssembler.transform(dataset);
        Dataset<Row> dataset1 = transform.select("label", "features");
        //dataset1.show();

        // Train and Test Split
        Dataset<Row>[] train_test_data = dataset1.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> train = train_test_data[0];
        Dataset<Row> test = train_test_data[1];

        // Create Model
        NaiveBayes nb = new NaiveBayes();
        nb.setSmoothing(1);
        NaiveBayesModel model = nb.fit(train);
        Dataset<Row> predictions = model.transform(test);

        // Evaluate Model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double evaluate = evaluator.evaluate(predictions);
        System.out.println("Accuracy = " +evaluate);

    }

    public static String[] toLowerList(List<String> data){
        List<String> res=new ArrayList<String>();
        for(String d:data){
            if(d.equals("Outcome")){
                res.add("label");
            }
            else{
                res.add(d.toLowerCase()+"_cat");
            }
        }
        String[] array = res.toArray(new String[res.size()]);
        return array;
    }
}
