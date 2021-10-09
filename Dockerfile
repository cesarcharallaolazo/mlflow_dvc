FROM python:3.9-slim
RUN apt update && apt install python-dev -y
RUN pip install mlflow psycopg2-binary boto3
EXPOSE 5000
COPY deploy-script.sh deploy-script.sh
RUN chmod +x deploy-script.sh
ENTRYPOINT [ "./deploy-script.sh" ]