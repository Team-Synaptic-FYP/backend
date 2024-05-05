FROM python:3.10-bullseye

WORKDIR /app

COPY ./requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=services.py

# Run services when the container launches
CMD ["python", "services.py"]



