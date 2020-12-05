#!/bin/sh
spark-submit \
--class com.madukubah.clustering.PostClustering \
--master local \
--deploy-mode client \
--executor-memory 1g \
--name analitic \
--conf "spark.app.id=analitic" \
--conf spark.driver.extraClassPath=jars/kafka-clients-0.10.0.1.jar:jars/spark-streaming-kafka-0-10_2.11-2.3.1.jar:jars/postgresql-42.2.12.jar \
target/analitic.jar

