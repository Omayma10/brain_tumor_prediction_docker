import sys
import csv
import subprocess
import requests
import json
import time
from requests.exceptions import ConnectionError

class BrainTumorPrediction:
    def __init__(self):
        pass

    def predict_tumor(self, data):
        max_retries = 5
        retry_num = 0
        while retry_num < max_retries:
            try:
                url = 'http://flask-server:8080/api/receive_values'
                data_send = {'values': data}

                response = requests.post(url, json=data_send)

                if response.status_code == 200:
                    print("Values sent successfully")
                    print(response.text)
                    break  # Break out of the loop if the request was successful
                else:
                    print("Failed to send values. Error:", response.text)
                    break  # Break out of the loop if there was a response other than 200
            except ConnectionError as e:
                print(f'Connection failed, retry {retry_num + 1}/{max_retries}')
                time.sleep(5)  # Wait for 5 seconds before retrying
                retry_num += 1

        if retry_num == max_retries:
            print("Failed to connect to Flask server after several retries.")

def main():
    input_labels = ['Variance:', 'Standard Deviation:', 'Entropy:', 'Skewness:', 'Kurtosis:',
                    'Contrast:', 'Energy:', 'ASM:', 'Homogeneity:', 'Dissimilarity:']
    input_data = []

    for label in input_labels:
        try:
            value = input(label)
            input_data.append(float(value))
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
            return

    predictor = BrainTumorPrediction()
    predictor.predict_tumor(input_data)

if __name__ == '__main__':
    main()
