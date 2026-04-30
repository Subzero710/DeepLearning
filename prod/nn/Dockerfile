FROM maven:3.9.9-eclipse-temurin-21-alpine AS builder
WORKDIR /project
COPY pom.xml /project/pom.xml
COPY src /project/src
RUN mvn clean package -DskipTests

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
RUN addgroup --system web && adduser -S -s /bin/false -G web app
COPY --from=builder /project/target/ecg200-front-0.0.1-SNAPSHOT.jar /app/app.jar
RUN chown app:web /app/app.jar
USER app
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
