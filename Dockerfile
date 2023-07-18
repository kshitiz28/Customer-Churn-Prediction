# Set base image
FROM python:3.9-slim
# Set directory for the project file and change to that directory
WORKDIR /Mlops
# Copy the current folder project to current docker project directory
COPY . .

EXPOSE 5000
# Install all the rquirements
RUN pip install -r Requirements.txt
RUN python customerchurnprediction.py
#Run the main python file with cmd command
# CMD [ "mlflow","ui" "--host" ]
ENTRYPOINT mlflow ui --host="0.0.0.0" --port="5000"