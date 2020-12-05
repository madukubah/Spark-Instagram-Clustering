package com.madukubah.analitic.clustering;

import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.monotonically_increasing_id;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import com.madukubah.analitic.config.Config;

public class PostClustering {
	private boolean verbose = false;
	private SparkSession ssc; 
	
	public PostClustering( SparkSession ssc ) 
	{
		this.ssc = ssc;
	}
	
	public class ClusterError implements Serializable {
		public int getCluster() {
			return cluster;
		}
		public void setCluster(int cluster) {
			this.cluster = cluster;
		}
		public double getValue() {
			return value;
		}
		public void setValue(double value) {
			this.value = value;
		}
		private int cluster ;
		private double value ;
		
		public ClusterError(int cluster, double value) {
			super();
			this.cluster = cluster;
			this.value = value;
		}
		public ClusterError()
		{

		}
	}
	public void findBestK(String table, int start, int end) 
	{
		
//		Dataset<Row> wsse= this.ssc.createDataFrame(Arrays.asList(
//				new ClusterError(2,11777.945015943516 ),
//				new ClusterError(3,10093.533018901428 ),
//				new ClusterError(4,9983.057873216738 ),
//				new ClusterError(5,9101.143749900859 ),
//				new ClusterError(6,8444.184134499132 ),
//				new ClusterError(7,8602.77459171629 ),
//				new ClusterError(8,7285.5776709409865 ),
//				new ClusterError(9,7859.377099706074 ),
//				new ClusterError(10,7235.805825230768 ),
//				new ClusterError(11,6620.923155611148 ),
//				new ClusterError(12,6276.053787386599 ),
//				new ClusterError(13,6422.4974770691315 ),
//				new ClusterError(14,5726.147351674693 ),
//				new ClusterError(15,5715.694294272607 ),
//				new ClusterError(16,5412.168085583254 ),
//				new ClusterError(17,5425.502601892976 ),
//				new ClusterError(18,5179.444238817171 ),
//				new ClusterError(19,5076.7721455895835 ),
//				new ClusterError(20,5403.548306187716)
//				), ClusterError.class);
//		wsse.write()
//		.mode(SaveMode.Overwrite)
//		  .format("jdbc")
//		  .option("url", Config.DB_NAME)
//		  .option("dbtable", "wsse")
//		  .option("user", Config.DB_USERNAME)
//		  .option("password", Config.DB_USERPASSWORD)
//		  .save();
//		
//		Dataset<Row> evaluators= this.ssc.createDataFrame(Arrays.asList(
//				new ClusterError(2,0.334907535477616 ),
//				new ClusterError(3,0.31483204795566483 ),
//				new ClusterError(4,0.3081159222361737 ),
//				new ClusterError(5,0.3018531186253041 ),
//				new ClusterError(6,0.3325589802290322 ),
//				new ClusterError(7,0.29577425578608024 ),
//				new ClusterError(8,0.40816626748262674 ),
//				new ClusterError(9,0.3756067954013196 ),
//				new ClusterError(10,0.4100829556652956 ),
//				new ClusterError(11,0.3918236040198202 ),
//				new ClusterError(12, 0.4110456337951909 ),
//				new ClusterError(13, 0.38791168370188545 ),
//				new ClusterError(14, 0.4343632091729015 ),
//				new ClusterError(15, 0.44004978420603325 ),
//				new ClusterError(16, 0.42916397567067843 ),
//				new ClusterError(17, 0.42966425073299536 ),
//				new ClusterError(18, 0.4583512903014146 ),
//				new ClusterError(19, 0.4628263179036934 ),
//				new ClusterError(20, 0.4028758695893859 )
//				), ClusterError.class);
//		evaluators.write()
//		.mode(SaveMode.Overwrite)
//		  .format("jdbc")
//		  .option("url", Config.DB_NAME)
//		  .option("dbtable", "evaluator")
//		  .option("user", Config.DB_USERNAME)
//		  .option("password", Config.DB_USERPASSWORD)
//		  .save();

		
		Properties connectionProperties = new Properties();
		connectionProperties.put("user", Config.DB_USERNAME );
		connectionProperties.put("password", Config.DB_USERPASSWORD);
		Dataset<Row> jdbcDF = this.ssc.read()
				  .jdbc(Config.DB_NAME, table, connectionProperties);
		
		jdbcDF.createOrReplaceTempView("posts");
		
		Dataset<Row> sampleDf = createSample( jdbcDF );
		Dataset<Row> dfwordCount = createWordCount( sampleDf.select("desc_image" ) );
		
		Dataset<Row> inputData = CreateInputData( sampleDf, dfwordCount ); 
		
		KMeans kMeans = new KMeans();
		
		List<ClusterError> wssseList = new ArrayList<>();
		List<ClusterError> evaluatorsList = new ArrayList<>();
		
		for( int noOfCluster =start; noOfCluster <=end; noOfCluster++ )
		{
			kMeans.setK(noOfCluster);
			KMeansModel model = kMeans.fit(inputData);
			
			Dataset<Row> predictions =model.transform(inputData);

			
			ClusteringEvaluator evaluator = new ClusteringEvaluator();
			double error = model.computeCost(inputData);
			double evaluation = evaluator.evaluate( predictions );
			
			wssseList.add(new ClusterError( noOfCluster, error));
			evaluatorsList.add( new ClusterError( noOfCluster, evaluation) );
			
			System.out.println("K : " + noOfCluster );
			System.out.println( "WSSSE : " + error );
			System.out.println( "Silhouette with squared euclidian distance : " + evaluation );
		}
		
		Dataset<Row> wsse= this.ssc.createDataFrame( wssseList, ClusterError.class );
		wsse.write()
		.mode(SaveMode.Overwrite)
		  .format("jdbc")
		  .option("url", Config.DB_NAME)
		  .option("dbtable", "wsse")
		  .option("user", Config.DB_USERNAME)
		  .option("password", Config.DB_USERPASSWORD)
		  .save();
		
		Dataset<Row> evaluators= this.ssc.createDataFrame( evaluatorsList, ClusterError.class );
		evaluators.write()
		.mode(SaveMode.Overwrite)
		  .format("jdbc")
		  .option("url", Config.DB_NAME)
		  .option("dbtable", "evaluator")
		  .option("user", Config.DB_USERNAME)
		  .option("password", Config.DB_USERPASSWORD)
		  .save();
	}
	
	public void doClustering(String table, int noOfCluster ) 
	{
		Properties connectionProperties = new Properties();
		connectionProperties.put("user", Config.DB_USERNAME );
		connectionProperties.put("password", Config.DB_USERPASSWORD);
		Dataset<Row> jdbcDF = this.ssc.read()
				  .jdbc(Config.DB_NAME, table, connectionProperties);
		
		doClustering(jdbcDF, noOfCluster ); 
	}
	
	public void doClustering(Dataset<Row> jdbcDF, int noOfCluster ) 
	{
		jdbcDF.createOrReplaceTempView("posts");
		
		Dataset<Row> sampleDf = createSample( jdbcDF );
//		return;
		Dataset<Row> dfwordCount = createWordCount( sampleDf.select("desc_image" ) );
		
		Dataset<Row> inputData = CreateInputData( sampleDf, dfwordCount ); 
		
		KMeans kMeans = new KMeans();
		
		kMeans.setK(noOfCluster);
		KMeansModel model = kMeans.fit(inputData);
		
		Dataset<Row> predictions =model.transform(inputData);

		sampleDf = sampleDf.withColumn( "index", monotonically_increasing_id() );
		predictions = predictions.withColumn( "index1", monotonically_increasing_id() );
		sampleDf = sampleDf.join(predictions, sampleDf.col("index").equalTo(predictions.col("index1")) )
				.drop("index")
				.drop("index1")
				.drop("features")
				.withColumnRenamed("prediction", "cluster");
		
		sampleDf.drop("features")
			.drop("image_count")
			.write()
			.mode(SaveMode.Overwrite)
			  .format("jdbc")
			  .option("url", Config.DB_NAME)
			  .option("dbtable", "sample_posts")
			  .option("user", Config.DB_USERNAME)
			  .option("password", Config.DB_USERPASSWORD)
			  .save();
		
		if (verbose) {
			ClusteringEvaluator evaluator = new ClusteringEvaluator();
			double error = model.computeCost(inputData);
			double evaluation = evaluator.evaluate( predictions );
			
			System.out.println("Cluster center : ");
			Vector[] clusterCenters = model.clusterCenters();
			for( Vector v : clusterCenters ) { System.out.println( v ); }
			
			System.out.println("K : " + noOfCluster );
			System.out.println( "WSSSE : " + error );
			System.out.println( "Silhouette with squared euclidian distance : " + evaluation );
		}
		setClusterInfo( sampleDf );
	}
	
	private void setClusterInfo( Dataset<Row> df ) 
	{	
		
		System.out.println("setClusterInfo " );
		List<org.apache.spark.sql.types.StructField> listOfStructField = new ArrayList<org.apache.spark.sql.types.StructField>();
		
		listOfStructField.add(DataTypes.createStructField("cluster", DataTypes.IntegerType, true));
		listOfStructField.add(DataTypes.createStructField("word", DataTypes.StringType, true));
		
		StructType structType = DataTypes.createStructType(listOfStructField);
		
		df = df.select("cluster", "desc_image" );
		Dataset<Row> clusterWord = df.flatMap( ( FlatMapFunction<Row, Row> ) row -> { 
			int i = row.fieldIndex("desc_image");
			String descImg = row.getString( i );
			
			int clusterIndex = row.fieldIndex("cluster");
			int cluster = row.getInt(clusterIndex);  
			
			descImg = descImg.replaceAll("[^a-zA-Z\\s]", "").toLowerCase().trim();
			List<String> words = Arrays.asList( descImg.split(" ") );
			
			List<Row> rowList = new ArrayList<>();
			for( String word : words )
			{
				rowList.add( RowFactory.create( cluster, word ) );
			}
			return rowList.iterator() ; 
			}, RowEncoder.apply(structType) );
		
		//clusterWord.show(10);
		clusterWord = clusterWord
				.groupBy("cluster", "word")
				.agg(count("word").as("count"))
				;
		
		clusterWord = clusterWord.groupBy("cluster")
				.agg(
					org.apache.spark.sql.functions.collect_list("word").as("words"),
					org.apache.spark.sql.functions.collect_list("count").as("counts")
				)
				;

		clusterWord
			.write()
			.mode(SaveMode.Overwrite)
			  .format("jdbc")
			  .option("url", Config.DB_NAME)
			  .option("dbtable", "cluster_words")
			  .option("user", Config.DB_USERNAME)
			  .option("password", Config.DB_USERPASSWORD)
			  .save();
	}
	
	private Dataset<Row> CreateInputData( Dataset<Row> sampleDf, Dataset<Row> dfwordCount ) 
	{		
		if (verbose) {
			System.out.println("CreateInputData");
		}
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
		return modelInputData ;
	}
	private Dataset<Row> createSample( Dataset<Row> rows ) {
		if (verbose) {
			System.out.println("createSample");
		}
		Dataset<Row> dataset= rows
				.select("username", "desc_image" )
				.groupBy("username")
				.agg(count("desc_image").as("image_count"))
				.withColumnRenamed("username", "username_x")
				;
		
//		create accounts
		createUserAccount( rows );
		
		Dataset<Row> filteredDf = dataset.filter("image_count >= 25"); 
		Dataset<Row> sampleDf = filteredDf
				.join(rows, rows.col("username").equalTo(filteredDf.col("username_x")), "left")
				;
		
		sampleDf = sampleDf.drop( sampleDf.col("username_x") )
				.filter( "LENGTH( TRIM( desc_image  )) > 0" )
				;
		
		createSampleUserAccount( sampleDf );
		return sampleDf;
	}
		
	private void createSampleUserAccount( Dataset<Row> rows ) {
		if (verbose) {
			System.out.println("createUserAccount");
		}
		
		Dataset<Row> accounts = rows;
		accounts = accounts
				.drop( accounts.col("desc_image") )
				.drop( accounts.col("source_image") )
				.drop( accounts.col("created_at") )
				.drop( accounts.col("updated_at") )
				.drop( accounts.col("id") )
				.distinct();
		
		accounts
			.write()
			.mode(SaveMode.Overwrite)
			  .format("jdbc")
			  .option("url", Config.DB_NAME)
			  .option("dbtable", "sample_accounts")
			  .option("user", Config.DB_USERNAME)
			  .option("password", Config.DB_USERPASSWORD)
			  .save();
		
	}
	private void createUserAccount( Dataset<Row> rows ) {
		if (verbose) {
			System.out.println("createUserAccount");
		}
		
		Dataset<Row> accounts = rows;
		accounts = accounts
				.drop( accounts.col("desc_image") )
				.drop( accounts.col("source_image") )
				.drop( accounts.col("created_at") )
				.drop( accounts.col("updated_at") )
				.drop( accounts.col("id") )
				.distinct();
		
		accounts
			.write()
			.mode(SaveMode.Overwrite)
			  .format("jdbc")
			  .option("url", Config.DB_NAME)
			  .option("dbtable", "accounts")
			  .option("user", Config.DB_USERNAME)
			  .option("password", Config.DB_USERPASSWORD)
			  .save();
		
	}
	
	private Dataset<Row> createWordCount( Dataset<Row> rowStrings ) 
	{
		if (verbose) {
			System.out.println("createWordCount");
		}
		
		Dataset<String> dfword = rowStrings.flatMap(
				( FlatMapFunction<Row, String>  ) s -> {
					String sentence = s.mkString();
					sentence = sentence.replaceAll("[^a-zA-Z\\s]", "").toLowerCase().trim();
					return Arrays.asList(  sentence.split(" ") ).iterator();
				}
				 ,Encoders.STRING() );
				
				
		dfword = dfword
				.map( ( MapFunction<String, String> ) word -> word.trim(), 
						Encoders.STRING())
				.filter("LENGTH( TRIM( value  )) > 0");
				
		
		Dataset<Row> dfwordCount = dfword
				.groupBy("value")
				.agg(count("value").as("count"))
				;
		
		dfwordCount = dfwordCount.withColumn( "id", monotonically_increasing_id() );
		
		dfwordCount
			.write()
			.mode(SaveMode.Overwrite)
			  .format("jdbc")
			  .option("url", Config.DB_NAME)
			  .option("dbtable", "word_counts")
			  .option("user", Config.DB_USERNAME)
			  .option("password", Config.DB_USERPASSWORD)
			  .save();
		
		dfwordCount = dfwordCount.withColumnRenamed("value", "desc_image");
		
		return dfwordCount ;
	}
	
	/**
	 * Set the verbose mode.
	 * @param verbose
	 * @return this object
	 */
	public PostClustering setVerbose(boolean verbose) {
		this.verbose = verbose;
		return this;
	}
	
	@SuppressWarnings({ "resource" })
	public static void main(String[] args)
	{
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		Logger.getLogger("org.apache").warn("pos clustering");
		SparkSession spark = SparkSession.builder().appName("analitic").master("local[*]").getOrCreate();
		PostClustering postClustering = new PostClustering( spark );
//		postClustering.findBestK("posts", 2, 20);
		postClustering.setVerbose(false).doClustering("posts", 8);
		spark.close();
	}
}
