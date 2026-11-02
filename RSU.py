import threading
from socket import *
import random

DROP_PROB = 0.3  # Ποσοστό Απωλειών %

# Server για V2X clients (δέχεται μηνύματα στην θύρα 12345)
def server_for_V2X_clients():

    # Socket για λήψη V2X message από V2X clients
    V2X_to_broker_socket = socket(AF_INET, SOCK_DGRAM)
    V2X_to_broker_socket.bind(("localhost", 12345))   
    print("[INFO] UDP server for V2X messages started")

    client_handler = threading.Thread(target=handle_V2X_clients, args=(V2X_to_broker_socket,))
    client_handler.start()

# Λήψη μηνυμάτων από τους V2X clients
def handle_V2X_clients(V2X_to_broker_socket):

    # Socket για αποστολή V2X message στο ego vehicle
    broker_to_ego_socket = socket(AF_INET, SOCK_DGRAM)
    broker_to_ego_socket.connect(("localhost", 5555))

    # Αναμονή μέχρι την λήψη V2X message από τους V2X clients
    while True:
        data = V2X_to_broker_socket.recv(4096)

        if random.random() > DROP_PROB:
            broker_to_ego_socket.send(data)

if __name__ == "__main__":
        
    server_for_V2X_clients()