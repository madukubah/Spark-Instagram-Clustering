package com.madukubah.analitic;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;

import com.madukubah.analitic.clustering.PostClustering;

public class Main {

	public static void main(String[] args) {
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		String mode = "clustering";
		Boolean verbose = false;
		int k = 8;
		
		Options options = new Options();
		options.addOption("mode", true, "clustering");
		options.addOption("range", true, "tool");
		options.addOption("k", true, "8" );
		
		CommandLineParser clparser = new PosixParser();
		CommandLine cm;
		try	
		{
			SparkSession spark = SparkSession.builder().appName("analitic").master("local[*]").getOrCreate();
			PostClustering postClustering = new PostClustering( spark );
			cm = clparser.parse(options, args);
			if (cm.hasOption("mode")) {
				mode = cm.getOptionValue("mode");
			}
			switch( mode )
			{
				case "clustering" :
					System.out.println( "clustering" );
					if (cm.hasOption("k")) {
						k = Integer.parseInt( cm.getOptionValue("k") ) ;
						System.out.println( "K = " + k );
					}
					if (cm.hasOption("verbose")) {
						verbose = true;
					}
					postClustering.setVerbose(true).doClustering("posts", k );
					break;
				
				case "searchingK" :
					System.out.println( "searchingK " );
//					postClustering.findBestK("posts", 2, 20);
					break;
			}
			spark.close();
		} catch (ParseException e) {
			e.printStackTrace();
		}
		
		
	}
	
}
