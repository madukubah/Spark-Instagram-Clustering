#!/bin/sh
spark-submit \
--master local \
--deploy-mode client \
--executor-memory 1g \
--name analitic \
--conf "spark.app.id=analitic" \
--conf spark.driver.extraClassPath=/home/madukubah/eclipse-workspace/analitic/jars/kafka-clients-0.10.0.1.jar:/home/madukubah/eclipse-workspace/analitic/jars/spark-streaming-kafka-0-10_2.11-2.3.1.jar:/home/madukubah/eclipse-workspace/analitic/jars/postgresql-42.2.12.jar \
 /home/madukubah/eclipse-workspace/analitic/target/analitic-0.0.1-SNAPSHOT.jar --mode=clustering --k=8


