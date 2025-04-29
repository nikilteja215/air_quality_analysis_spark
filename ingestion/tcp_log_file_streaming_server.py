import socket
import time
import csv
import os

# TCP Server Configuration
TCP_HOST = 'localhost'
TCP_PORT = 9999

# File Path
CSV_FILE_PATH = '/workspaces/air_quality_analysis_spark/ingestion/data/pending/AirQualityUCI.csv'


def create_tcp_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((TCP_HOST, TCP_PORT))
    server_socket.listen(5)
    print(f"✅ TCP server listening on {TCP_HOST}:{TCP_PORT}...")
    return server_socket


def send_data_to_client(client_socket):
    print(f"📄 Reading from: {CSV_FILE_PATH}")

    with open(CSV_FILE_PATH, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)  # Skip header line

        for row in reader:
            try:
                date = row[0]
                time_str = row[1]
                no2 = row[11]
                temperature = row[12]
                humidity = row[13]

                if not (date and time_str and no2 and temperature and humidity):
                    continue  # Skip if any field missing

                timestamp = f"{date} {time_str}"
                clean_line = f"{timestamp},Region1,{no2},{temperature},{humidity}"

                # Only send if line is clean
                client_socket.send((clean_line + '\n').encode('utf-8'))
                print(f"Sent: {clean_line}")

                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Skipping bad row: {e}")
                continue


def accept_connections(server_socket):
    while True:
        print("Waiting for new client connection...")
        client_socket, client_address = server_socket.accept()
        print(f"✅ Client connected: {client_address}")

        send_data_to_client(client_socket)

        client_socket.close()
        print(f"Closed connection with client: {client_address}")


def start_server():
    server_socket = create_tcp_server()
    accept_connections(server_socket)


if __name__ == '__main__':
    start_server()
