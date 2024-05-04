FROM python:3.10.0-alpine

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Make port 80 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=services.py

# Run app.py when the container launches
CMD ["python", "services.py"]



