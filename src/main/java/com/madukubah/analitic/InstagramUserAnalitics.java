package com.madukubah.analitic;

import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.monotonically_increasing_id;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
//import org.apache.log4j.Logger;
//import org.apache.log4j.Level;
//import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes; 


/**
 * SPARK SQL
 */
public class InstagramUserAnalitics 
{
	@SuppressWarnings({ "resource" })
	public static void main(String[] args)
	{
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder().appName("testingSQL").master("local[*]").getOrCreate();
		
		Properties connectionProperties = new Properties();
		connectionProperties.put("user", "madukubah");
		connectionProperties.put("password", "Alan!234");
		Dataset<Row> jdbcDF2 = spark.read()
		  .jdbc("jdbc:postgresql://localhost:5432/instanalitic", "posts", connectionProperties);
		
		
		jdbcDF2.createOrReplaceTempView("posts");
		
		Dataset<Row> df= jdbcDF2;
		
		Dataset<Row> dataset= df
				.select("username", "desc_image" )
				.groupBy("username")
				.agg(count("desc_image").as("image_count"))
				.withColumnRenamed("username", "username_x")
				;
		Dataset<Row> filteredDf = dataset.filter("image_count >= 25"); 
		
		Dataset<Row> sampleDf = filteredDf
				.join(df, df.col("username").equalTo(filteredDf.col("username_x")), "left")
				;
		
		sampleDf = sampleDf.drop( sampleDf.col("username_x") )
				.filter( "LENGTH( TRIM( desc_image  )) > 0" )
				;
	
		Dataset<String> dfword = sampleDf.select("desc_image" ).flatMap(
				( FlatMapFunction<Row, String>  ) s -> {
					String sentence = s.mkString();
					sentence = sentence.replaceAll("[^a-zA-Z\\s]", "").toLowerCase().trim();
					return Arrays.asList(  sentence.split(" ") ).iterator();
				}
//				 new FlatMapFunction<Row, String>()
//				 {
//					@Override
//					public Iterator<String> call(Row s) throws Exception {
//						String sentence = s.mkString();
//						sentence = sentence.replaceAll("[^a-zA-Z\\s]", "").toLowerCase().trim();
//						return Arrays.asList(  sentence.split(" ") ).iterator();
//					}
//				 }
				 ,Encoders.STRING() );
				
				
		dfword = dfword
				.map( ( MapFunction<String, String> ) word -> word.trim(), 
						Encoders.STRING())
				.filter("LENGTH( TRIM( value  )) > 0");
				
//				filter( word -> word.trim().length() > 0 );
		
		Dataset<Row> dfwordCount = dfword
				.groupBy("value")
				.agg(count("value").as("word_count"))
				;
		
		dfwordCount
			.write()
			.mode(SaveMode.Overwrite)
			  .format("jdbc")
			  .option("url", "jdbc:postgresql://localhost:5432/instanalitic")
			  .option("dbtable", "word_counts")
			  .option("user", "madukubah")
			  .option("password", "Alan!234")
			  .save();
		
		dfwordCount = dfwordCount.withColumnRenamed("value", "desc_image");
		
		Tokenizer tokenizer = new Tokenizer()
				.setInputCol("desc_image")
				.setOutputCol("tokens");
		sampleDf = tokenizer.transform(sampleDf);
		dfwordCount = tokenizer.transform(dfwordCount);
		CountVectorizer countVectorizer = new CountVectorizer()
				.setInputCol("tokens")
				.setOutputCol("features")
				; 
		sampleDf = countVectorizer.fit(dfwordCount).transform(sampleDf);
		
		
		Dataset<Row> modelInputData = sampleDf.select("features");
		
		KMeans kMeans = new KMeans();
		List<Double> wssse = new ArrayList<>();
		List<Double> evaluators = new ArrayList<>();
		for( int noOfCluster =8; noOfCluster <=8; noOfCluster++ )
		{
			kMeans.setK(noOfCluster);
			KMeansModel model = kMeans.fit(modelInputData);
			
			Dataset<Row> predictions =model.transform(modelInputData);

			//==============================================
			spark.udf().register("set_cluster", (String cluster ) -> { 
				return cluster;
			}, DataTypes.IntegerType );
			
			sampleDf = sampleDf.withColumn( "index", monotonically_increasing_id() );
			predictions = predictions.withColumn( "index1", monotonically_increasing_id() );
			sampleDf = sampleDf.join(predictions, sampleDf.col("index").equalTo(predictions.col("index1")) )
					.drop("index")
					.drop("index1")
					.drop("features")
					.withColumnRenamed("prediction", "cluster")
					;
			
			sampleDf.drop("features")
				.drop("image_count")
				.write()
				.mode(SaveMode.Overwrite)
				  .format("jdbc")
				  .option("url", "jdbc:postgresql://localhost:5432/instanalitic")
				  .option("dbtable", "sample_posts")
				  .option("user", "madukubah")
				  .option("password", "Alan!234")
				  .save();
			//==============================================
			System.out.println("Cluster center : ");
//			Vector[] clusterCenters = model.clusterCenters();
//			for( Vector v : clusterCenters ) { System.out.println( v ); }
			
			ClusteringEvaluator evaluator = new ClusteringEvaluator();
			double error = model.computeCost(modelInputData);
			double evaluation = evaluator.evaluate( predictions );
			
			wssse.add(error);
			evaluators.add(evaluation);
			
			System.out.println("K : " + noOfCluster );
			System.out.println( "WSSSE : " + error );
			System.out.println( "Silhouette with squared euclidian distance : " + evaluation );
		}
		
		System.out.println( wssse );
		System.out.println( evaluators );
//		Scanner scanner = new Scanner(System.in);
//		scanner.nextLine(  );
		spark.close();
	}
}





















