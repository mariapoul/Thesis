from socket import *
import json
import carla
import time

# Socket για αποστολή V2X message στον broker
V2X_to_broker_socket = socket(AF_INET, SOCK_DGRAM)
V2X_to_broker_socket.connect(("localhost", 12345))

global total_sent
total_sent = 0

# DENM μήνυμα (μήνυμα πεζού)
def send_message_from_walker(walker, frame):

    global total_sent

    DENM = {

        "message_type": "DENM",
        "frame":        frame,
        "timestamp":    time.time(),
        "id":           walker.id,
        "object_type":  walker.type_id,
        "location": {
            "x":        walker.get_transform().location.x,
            "y":        walker.get_transform().location.y,
            "z":        walker.get_transform().location.z
        },
        "rotation": {
            "pitch":    walker.get_transform().rotation.pitch,
            "yaw":      walker.get_transform().rotation.yaw,
            "roll":     walker.get_transform().rotation.roll
        },
        "velocity":     walker.get_velocity().x
    }

    # Μετατροπή DENM μηνύματος στην κατάλληλη μορφή και αποστολή του στον broker
    json_string = json.dumps(DENM) # Μετατροπή Python Dictonary σε JSON string
    message = json_string.encode() # Μετατροπή JSON string σε bytes
    total_sent += 1
    V2X_to_broker_socket.send(message)

# CAM μήνυμα (μήνυμα οχήματος)
def send_message_from_vehicle(vehicle, frame):

    global total_sent

    CAM = {

        "message_type":   "CAM",
        "frame":          frame,
        "timestamp":      time.time(),
        "id":             vehicle.id,
        "object_type":    vehicle.type_id,
        "location": {
            "x":          vehicle.get_transform().location.x,
            "y":          vehicle.get_transform().location.y,
            "z":          vehicle.get_transform().location.z
        },
        "rotation": {
            "pitch":      vehicle.get_transform().rotation.pitch,
            "yaw":        vehicle.get_transform().rotation.yaw,
            "roll":       vehicle.get_transform().rotation.roll
        },
        "velocity":       vehicle.get_velocity().x,
        "acceleration":   vehicle.get_acceleration().x,
        "vehicle_length": vehicle.bounding_box.extent.x * 2,
        "vehicle_width":  vehicle.bounding_box.extent.y * 2,
        "vehicle_height": vehicle.bounding_box.extent.z * 2
    }

    # Μετατροπή CAM μηνύματος στην κατάλληλη μορφή και αποστολή του στον broker
    json_string = json.dumps(CAM) # Μετατροπή Python Dictonary σε JSON string
    message = json_string.encode() # Μετατροπή JSON string σε bytes
    total_sent += 1
    V2X_to_broker_socket.send(message)

# SPaT μήνυμα (μήνυμα φαναριού)
def send_message_from_traffic_light(traffic_light, frame):

    global total_sent

    SPaT = {
    
        "message_type":       "SPaT",
        "frame":              frame,
        "timestamp":          time.time(),
        "id":                 traffic_light.id,
        "object_type":        traffic_light.type_id,
        "location": {
            "x":              traffic_light.get_transform().location.x,
            "y":              traffic_light.get_transform().location.y,
            "z":              traffic_light.get_transform().location.z
        },
        "rotation": {
            "pitch":          traffic_light.get_transform().rotation.pitch,
            "yaw":            traffic_light.get_transform().rotation.yaw,
            "roll":           traffic_light.get_transform().rotation.roll
        },
        "state":              traffic_light.get_state(),
        "time_to_turn_green": traffic_light.get_green_time()-traffic_light.get_elapsed_time(),
        "time_to_turn_red":   traffic_light.get_red_time()-traffic_light.get_elapsed_time()
    }

    # Μετατροπή SPaT μηνύματος στην κατάλληλη μορφή και αποστολή του στον broker
    json_string = json.dumps(SPaT) # Μετατροπή Python Dictonary σε JSON string
    message = json_string.encode() # Μετατροπή JSON string σε bytes
    total_sent += 1
    V2X_to_broker_socket.send(message)

# IVI μήνυμα (μήνυμα σήματος)
def send_message_from_traffic_sign(traffic_sign, frame):

    global total_sent

    IVI = {

        "message_type": "IVI",
        "frame":        frame,
        "timestamp":    time.time(),
        "id":           traffic_sign.id,
        "object_type":  traffic_sign.type_id,
        "location": {
            "x":        traffic_sign.get_transform().location.x,
            "y":        traffic_sign.get_transform().location.y,
            "z":        traffic_sign.get_transform().location.z
        },
        "rotation": {
            "pitch":    traffic_sign.get_transform().rotation.pitch,
            "yaw":      traffic_sign.get_transform().rotation.yaw,
            "roll":     traffic_sign.get_transform().rotation.roll
        }
    }

    # Μετατροπή IVI μηνύματος στην κατάλληλη μορφή και αποστολή του στον broker
    json_string = json.dumps(IVI) # Μετατροπή Python Dictonary σε JSON string
    message = json_string.encode() # Μετατροπή JSON string σε bytes
    total_sent += 1
    V2X_to_broker_socket.send(message)

# Πρόσβαση σε όλους τους clients εντός μιας ακτίνας του ego vehicle
if __name__ == "__main__":

    client = carla.Client('localhost', 2000)
    world = client.get_world()

    vehicles = list(world.get_actors().filter('vehicle.*'))
    while not vehicles:
        print("[INFO] Waiting for ego vehicle to spawn\n")
        time.sleep(0.5)
        vehicles = list(world.get_actors().filter('vehicle.*'))
    ego_vehicle = vehicles[-1]

    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    traffic_signs = list(world.get_actors().filter('traffic.stop')) + list(world.get_actors().filter('traffic.yield'))

    # Αναμονή μέχρι την λήψη frame από τον broker
    while True:

        if world.get_actors().find(ego_vehicle.id) is None:
            print("\nEgo vehicle destroyed, stopping V2X client")
            print(f"V2X Messages sent: {total_sent}")
            input("Press Enter to close...")
            break

        frame = world.wait_for_tick().frame
        print(f"===== FRAME {frame} =======")

        # DENM μήνυμα (κάθε 10 frames, 0.5s)
        if frame % 10 == 0:
            walkers = world.get_actors().filter('walker.*')
            for walker in walkers:
                if walker.get_transform().location.distance(ego_vehicle.get_transform().location)<30:
                    send_message_from_walker(walker, frame)
                    print(f"DENM message sent from id={walker.id}, type={walker.type_id}")

        # CAM μήνυμα (κάθε 10 frames, 0.5s)
        if frame % 10 == 0:
            vehicles = world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                if vehicle.id == ego_vehicle.id:
                    continue

                if vehicle.get_transform().location.distance(ego_vehicle.get_transform().location)<30:
                    send_message_from_vehicle(vehicle, frame)
                    print(f"CAM message sent from id={vehicle.id}, type={vehicle.type_id}")

        # SPAT μήνυμα (κάθε 30 frames, 1.5s)
        if frame % 30 == 0:
            for traffic_light in traffic_lights:
                if traffic_light.get_transform().location.distance(ego_vehicle.get_transform().location)<45: 
                    send_message_from_traffic_light(traffic_light, frame)
                    print(f"SPaT message sent from id={traffic_light.id}, type={traffic_light.type_id}")

        # IVI μήνυμα (κάθε 40 frames, 2s)
        if frame % 40 == 0:
            for traffic_sign in traffic_signs:
                if traffic_sign.get_transform().location.distance(ego_vehicle.get_transform().location)<20:
                    send_message_from_traffic_sign(traffic_sign, frame)
                    print(f"IVI message sent from id={traffic_sign.id}, type={traffic_sign.type_id}")

        print("=========================\n")