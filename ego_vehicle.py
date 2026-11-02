import carla
import random
import numpy as np
import cv2 
import time
import math
import open3d as o3d
import warnings
import subprocess
import sys
import json
from socket import *

# Φιλτράρισμα στην εμφάνιση των warnings για την χρήση των εντολών στις γραμμές 21 και 141 του κώδικα
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Δήλωση των περιορισμών ταχύτητας
PREFERRED_SPEED = 20 
SPEED_THRESHOLD = 2

global vehicle

# Μεταβλητές για Semantic LIDAR 
global lidar_data             # δήλωση global μεταβλητής ώστε να έχουμε πρόσβαση στα δεδομένα του Semantic LIDAR και συγκεκριμένα στην μεταβλητή object_tag 
lidar_data = np.zeros((0,6)) 
lidar_added = False           # δήλωση μεταβλητής για να ξέρουμε πότε προστέθηκαν δεδομένα στον Semantic LIDAR 

# Μεταβλητές για τους Obstacle Detector
global vehicle_detected # δήλωση global μεταβλητής ώστε να γνωρίζουμε πότε ανιχνεύθηκε κάποιο όχημα 
vehicle_detected = False
global walker_detected # δήλωση global μεταβλητής ώστε να γνωρίζουμε πότε ανιχνεύθηκε κάποιος πεζός 
walker_detected = False

# Μεταβλητές για Collision Detector
hit_walkers = set()
hit_vehicles = set()

# !!! V2X !!!
# Μεταβλητές για DENM/CAM
v2x_flags = {
    "walker": False,
    "vehicle": False
}

v2x_hold = {
    "walker": 0,
    "vehicle": 0
}

V2X_HOLD_FRAMES = 10 # 0.5s

# Μεταβλητές για V2X
global total_received 
total_received = 0

global latencies
latencies = []
# !!!!!!!!!!!

# Δημιουργία αξόνων στο Open3D Visualizer
def add_open3d_axis(vis):

    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],   # Σημείο 0, αρχή των αξόνων
        [1.0, 0.0, 0.0],   # Σημείο 1, τέλος του άξονα x
        [0.0, 1.0, 0.0],   # Σημείο 2, τέλος του άξονα y
        [0.0, 0.0, 1.0]])) # Σημείο 3, τέλος του άξονα z
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],   # Ένωση σημείου 0 με σημείο 1, σχηματισμός άξονα x
        [0, 2],   # Ένωση σημείου 0 με σημείο 2, σχηματισμός άξονα y
        [0, 3]])) # Ένωση σημείου 0 με σημείο 3, σχηματισμός άξονα z
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],   # Κόκκινο χρώμα για τον άξονα x (R)
        [0.0, 1.0, 0.0],   # Πράσινο χρώμα για τον άξονα y (G)
        [0.0, 0.0, 1.0]])) # Μπλε χρώμα για τον άξονα z (B)
    vis.add_geometry(axis)

# Μετατροπή 3D σημείων από την κάμερα σε pixel στην εικόνα
def build_projection_matrix(w, h, fov):

    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# Μετατροπή 3D σημείου, σε 2D σημείο στην εικόνα της κάμερας
def get_image_point(loc, K, w2c):

    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return tuple(map(int, point_img[0:2]))

# Δημιουργία εικόνας RGB κάμερας (image -> ύψος (height) x πλάτος (width) x 4 (R, G, B, A για το κάθε pixel)) 
def rgb_callback(image, data_dict):
    
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# Λήψη δεδομένων Semantic LiDAR (point_cloud -> συντεταγμένες x, y, z, κατεύθυνση ακτίνας (cos_inc_angle), ID αντικειμένου (object_idx), κατηγορία αντικειμένου (object_tag)) και αντιστοίχιση σημείων σε χρώματα
def semantic_lidar_callback(point_cloud, point_list):

    global lidar_data
    
    semantic_tag_colors = {
        0:  (0, 0, 0),       # Unlabeled	
        1:  (128, 64, 128),  # Roads
        2:  (244, 35, 232),  # SideWalks	
        3:  (70, 70, 70),    # Building		
        4:  (102, 102, 156), # Wall
        5:  (190, 153, 153), # Fence	
        6:  (153, 153, 153), # Pole	
        7:  (250, 170, 30),  # TrafficLight	
        8:  (220, 220, 0),   # TrafficSign		
        9:  (107, 142, 35),  # Vegetation	
        10: (152, 251, 152), # Terrain	
        11:	(70, 130, 180),	 # Sky	
        12:	(220, 20, 60),   # Pedestrian	
        13:	(255, 0, 0),     # Rider		
        14:	(0, 0, 142),	 # Car
        15:	(0, 0, 70),	     # Truck	
        16:	(0, 60, 100),    # Bus	
        17:	(0, 80, 100),    # Train		
        18:	(0, 0, 230),     # Motorcycle	
        19:	(119, 11, 32),   # Bicycle		
        20:	(110, 190, 160), # Static	
        21:	(170, 120, 50),  # Dynamic	
        22:	(55, 90, 80),    # Other	
        23:	(45, 60, 150),   # Water	
        24:	(157, 234, 50),  # RoadLine	
        25:	(81, 0, 81),	 # Ground	
        26:	(150, 100, 100), # Bridge	
        27:	(230, 150, 140), # RailTrack	
        28:	(180, 165, 180), # GuardRail
    }	

    dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('cos_inc_angle', 'f4'), ('object_idx', 'u4'), ('object_tag', 'u4')])
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype)) # αντιγραφή των σημείων του point_cloud σε μονοδιάστατο πίνακα
    data = np.stack([data['x'], data['y'], data['z'], data['cos_inc_angle'], data['object_idx'], data['object_tag']], axis=1) # πίνακας Nx6 (float), όπου κάθε γραμμή είναι ένα σημείο του point_cloud [x, y, z, cos_inc_angle, object_idx, object_tag]
    lidar_data = data

    object_tags = data[:, 5].astype(np.uint32) # παίρνουμε μόνο το object_tag και το μετατρέπουμε σε ακέραιο

    colors = np.array([semantic_tag_colors.get(tag, (255, 255, 255)) for tag in object_tags]) / 255.0 # αντιστοίχιση κάθε object_tag με το χρώμα του βάση του semantic_tag_colors

    points = data[:, :3] # πίνακας που συμπεριλαμβάνει τα σημεία
    points[:, 1] = -points[:, 1] # αντιστροφή του άξονα y για να ταυτίζεται το σύστημα του Unreal Engine/Open3D με το σύστημα του Carla
    point_list.points = o3d.utility.Vector3dVector(points) # ορίζουμε τα σημεία που θα εμφανιστούν στο Open3D Visualizer
    point_list.colors = o3d.utility.Vector3dVector(colors) # ορίζουμε τα χρώματα των σημείων που θα εμφανιστούν στο Open3D Visualizer

# Λήψη δεδομένων Obstacle Detector για ανίχνευση πεζών (μέσω του κράνους) (event -> αντικείμενο που ανιχνεύτηκε (other_actor), frame, απόσταση του αντικειμένου (distance))
def walker_obstacle_callback(event, data_dict, camera, k_mat):

    global walker_detected

    # Ανίχνευση πεζών (μέσω του κράνους)
    if event.other_actor.type_id.startswith('static.prop.bike helmet'):
        print(f"Detected walker with id={event.other_actor.id}, type={event.other_actor.type_id}")
        walker_detected = True
        data_dict['obstacle'].append({'type': event.other_actor.type_id, 'frame': event.frame})
    
    # Μετατροπή 3D συντεταγμένων του εμποδίου στο σύστημα της κάμερας
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Μετατροπή 3D σημείου της θέσης του εμποδίου, σε 2D σημείο στην εικόνα της κάμερας, δηλαδή σε pixel
    image_point = get_image_point(event.other_actor.get_transform().location, k_mat, world_2_camera)

    # Αν το εμπόδιο πέφτει μέσα στην εικόνα της κάμερας, τότε ζωγραφίζουμε έναν κόκκινο κύκλο 
    if 0<image_point[0]<image_w and 0<image_point[1]<image_h:
        cv2.circle(data_dict['rgb_image'], tuple(image_point), 10, (0, 0, 255), 3)

# Λήψη δεδομένων Obstacle Detector για ανίχνευση οχημάτων (event -> αντικείμενο που ανιχνεύτηκε (other_actor), frame, απόσταση του αντικειμένου (distance))
def vehicle_obstacle_callback(event, data_dict, camera, k_mat):

    global vehicle_detected

    # Ανίχνευση οχημάτων
    if event.other_actor.type_id.startswith('vehicle.'):
        print(f"Detected vehicle with id={event.other_actor.id}, type={event.other_actor.type_id}")
        vehicle_detected = True
        data_dict['obstacle'].append({'type': event.other_actor.type_id, 'frame': event.frame})
    
    # Μετατροπή 3D συντεταγμένων του εμποδίου στο σύστημα της κάμερας
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Μετατροπή 3D σημείου της θέσης του εμποδίου, σε 2D σημείο στην εικόνα της κάμερας, δηλαδή σε pixel
    image_point = get_image_point(event.other_actor.get_transform().location, k_mat, world_2_camera)

    # Αν το εμπόδιο πέφτει μέσα στην εικόνα της κάμερας, τότε ζωγραφίζουμε έναν κόκκινο κύκλο 
    if 0<image_point[0]<image_w and 0<image_point[1]<image_h:
        cv2.circle(data_dict['rgb_image'], tuple(image_point), 10, (0, 0, 255), 3)

# Λήψη δεδομένων Collision Detector για καταμέτρηση συγκρούσεων με πεζούς και οχήματα
def collision_callback(event, data_dict):

    if event.other_actor.type_id.startswith('walker.'):
        if event.other_actor.id not in hit_walkers:
            hit_walkers.add(event.other_actor.id)
            data_dict['walker_collision_count']+=1

    elif event.other_actor.type_id.startswith('vehicle.'):
        if event.other_actor.id not in hit_vehicles:
            hit_vehicles.add(event.other_actor.id)
            data_dict['vehicle_collision_count']+=1

    data_dict['collision']=True       

# Συνάρτηση διατήρησης σταθερής ταχύτητας
def maintain_speed(s):
    
    if s >= PREFERRED_SPEED:
        return 0
    elif s < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.7 
    else:
        return 0.4

# !!! V2X !!!
def handle_V2X_messages(data):

    global vehicle

    msg = json.loads(data.decode())

    msg_type = msg.get("message_type")
    object_location_x = msg.get("location", {}).get("x")
    object_location_y = msg.get("location", {}).get("y")
    object_yaw = msg.get("rotation", {}).get("yaw")
    sent_time = msg.get("timestamp")

    latency = time.time()-sent_time
    latencies.append(latency)

    if msg_type == "DENM":

        # Εύρεση κατεύθυνσης μεθεπόμενου waypoint
        current_wp = world.get_map().get_waypoint(vehicle.get_location())
        next_wp = current_wp.next(2.0)[0] # παίρνει το πρώτο waypoint που βρίσκεται τουλάχιστον 2 μέτρα μπροστά από το τωρινό (μεθεπόμενο waypoint)   
        next_wp_yaw = math.radians(next_wp.transform.rotation.yaw)

        # Vehicle coordinates (καρτεσιανές συντεταγμένες) βάση του μεθεπόμενου waypoint 
        dx = object_location_x - vehicle.get_transform().location.x
        dy = object_location_y - vehicle.get_transform().location.y
        local_x =  math.cos(next_wp_yaw) * dx + math.sin(next_wp_yaw) * dy
        local_y = -math.sin(next_wp_yaw) * dx + math.cos(next_wp_yaw) * dy

        # Έλεγχος εντός ενός ορθογωνίου βάση του μεθεπόμενου waypoint, ελέγχουμε εντός της λωρίδας μας 
        if 0<local_x<10 and abs(local_y)<1.8:
            print(f"\nDENM:\n{msg}\n")
            v2x_flags["walker"] = True
            v2x_hold["walker"] = V2X_HOLD_FRAMES

    elif msg_type == "CAM":

        # Εύρεση κατεύθυνσης μεθεπόμενου waypoint
        current_wp = world.get_map().get_waypoint(vehicle.get_location())
        next_wp = current_wp.next(2.0)[0] # παίρνει το πρώτο waypoint που βρίσκεται τουλάχιστον 2 μέτρα μπροστά από το τωρινό (μεθεπόμενο waypoint)   
        next_wp_yaw = math.radians(next_wp.transform.rotation.yaw)

        # Vehicle coordinates (καρτεσιανές συντεταγμένες) βάση του μεθεπόμενου waypoint 
        dx = object_location_x - vehicle.get_transform().location.x
        dy = object_location_y - vehicle.get_transform().location.y
        local_x =  math.cos(next_wp_yaw) * dx + math.sin(next_wp_yaw) * dy
        local_y = -math.sin(next_wp_yaw) * dx + math.cos(next_wp_yaw) * dy

        # Έλεγχος εντός ενός ορθογωνίου βάση του μεθεπόμενου waypoint, ελέγχουμε εντός της λωρίδας μας 
        if 0<local_x<13 and abs(local_y)<1.2:
            print(f"\nCAM:\n{msg}\n")
            v2x_flags["vehicle"] = True 
            v2x_hold["vehicle"] = V2X_HOLD_FRAMES

    elif msg_type == "SPaT":

        # Απόσταση μεταξύ σήματος και οχήματος 
        dx = object_location_x - vehicle.get_transform().location.x
        dy = object_location_y - vehicle.get_transform().location.y

        # Διαφορά γωνίας μεταξύ σήματος και οχήματος (είναι μπροστά μου?)
        vehicle_yaw = math.radians(vehicle.get_transform().rotation.yaw)    
        target_yaw = math.atan2(dy, dx)                                     
        angle_diff = target_yaw - vehicle_yaw                             
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        # Ανίχνευση φαναριού μπροστά και δεξιά από το όχημα και με διαφορά κατεύθυνσης 90 περίπου μοίρες
        if 0<angle_diff<math.pi/2 and abs(object_yaw-vehicle.get_transform().rotation.yaw-90)<10:
            print(f"\nSPaT:\n{msg}\n")

    elif msg_type == "IVI":

        # Απόσταση μεταξύ σήματος και οχήματος 
        dx = object_location_x - vehicle.get_transform().location.x
        dy = object_location_y - vehicle.get_transform().location.y

        # Διαφορά γωνίας μεταξύ σήματος και οχήματος (είναι μπροστά μου?)
        vehicle_yaw = math.radians(vehicle.get_transform().rotation.yaw)    
        target_yaw = math.atan2(dy, dx)                                     
        angle_diff = target_yaw - vehicle_yaw                             
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff)) 

        # Ανίχνευση σήματος μπροστά από το όχημα και με διαφορά κατεύθυνσης 90 περίπου μοίρες
        if abs(angle_diff)<math.pi/2 and abs(object_yaw-vehicle.get_transform().rotation.yaw-90)<5:
            print(f"\nIVI:\n{msg}\n")
# !!!!!!!!!!!

try:
    # Σύνδεση του client (Python API) με τον server (CARLA simulator)
    client = carla.Client('localhost', 2000) 

    # Πρόσβαση στον χώρο της προσομοίωσης 
    world = client.get_world() 

    # Ενεργοποίηση synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Εύρεση διαθέσιμων σημείων για την τοποθέτηση του οχήματος
    spawn_points = world.get_map().get_spawn_points()

    # Λίστα για όλους τους actors
    actors = []

    # Εύρεση Mini οχημάτων και επιλογή του πρώτου
    vehicle_bps = world.get_blueprint_library().filter('*mini*')
    vehicle_bp = vehicle_bps[0]

    # Τοποθέτηση του Mini οχήματος σε ένα από τα διαθέσιμα σημεία 
    start_point = spawn_points[30]
    #start_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, start_point)
    actors.append(vehicle)
                                        
    # Εκτύπωση διαστάσεων οχήματος
    print(f"Vehicle's length: {vehicle.bounding_box.extent.x * 2}, width: {vehicle.bounding_box.extent.y * 2} and height: {vehicle.bounding_box.extent.z * 2}\n")

    # Μέγιστη γωνία περιστροφής του κάθε τροχού
    for i, wheel in enumerate(vehicle.get_physics_control().wheels):
        print(f"Wheel {i}: max steer angle = {wheel.max_steer_angle:.2f}°")

    # Μετακίνηση του παραθύρου του προσομοιωτή στο πεδίο ορατότητας του οδηγού
    spectator = world.get_spectator()
    start_point.location.z = start_point.location.z+1
    spectator.set_transform(start_point)

    # Ρύθμιση RGB κάμερας και τοποθέτηση πάνω από το παρμπρίζ του οχήματος
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(x = 0.6, z = vehicle.bounding_box.extent.z * 2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to = vehicle)
    actors.append(camera)

    # Ρύθμιση Semantic LiDAR και τοποθέτηση στην οροφή του οχήματος
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('range', '100.0') # σε μέτρα
    lidar_bp.set_attribute('upper_fov', '40.0')
    lidar_bp.set_attribute('lower_fov', '-40.0')
    lidar_bp.set_attribute('channels', '64.0')
    lidar_bp.set_attribute('rotation_frequency', '20.0')
    lidar_bp.set_attribute('points_per_second', '500000') 
    lidar_init_trans = carla.Transform(carla.Location(z = vehicle.bounding_box.extent.z * 2 + 0.2))
    lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to = vehicle)
    actors.append(lidar)

    # Ρύθμιση Obstacle Detector για ανίχνευση οχημάτων και τοποθέτηση πάνω από το παρμπρίζ του οχήματος
    vehicle_obstacle_bp = world.get_blueprint_library().find('sensor.other.obstacle')
    vehicle_obstacle_bp.set_attribute('distance', '5.176') # ανίχνευση εμποδίων εντός 3.5 μέτρων μπροστά από το όχημα
    vehicle_obstacle_bp.set_attribute('hit_radius', '1.1') # 1.1 μέτρα πλάτος για τα οχήματα για να μην υπάρχει θέμα με ανίχνευση κατά την κίνηση 
    obstacle_init_trans = carla.Transform(carla.Location(x = 0.6, z = vehicle.bounding_box.extent.z * 2 - 0.2)) 
    vehicle_obstacle_sensor = world.spawn_actor(vehicle_obstacle_bp, obstacle_init_trans, attach_to = vehicle)
    actors.append(vehicle_obstacle_sensor)

    # Ρύθμιση Obstacle Detector για ανίχνευση πεζών (μέσω των κώνων) και τοποθέτηση πάνω από το παρμπρίζ του οχήματος
    walker_obstacle_bp = world.get_blueprint_library().find('sensor.other.obstacle')
    walker_obstacle_bp.set_attribute('distance', '5.176') # ανίχνευση εμποδίων εντός 3.5 μέτρων μπροστά από το όχημα
    walker_obstacle_bp.set_attribute('hit_radius', '1.9') # 1.9 μέτρα πλάτος για τους πεζούς για να προλαβαίνει το όχημα να σταματήσει 
    obstacle_init_trans = carla.Transform(carla.Location(x = 0.6, z = vehicle.bounding_box.extent.z * 2 + 0.2)) 
    walker_obstacle_sensor = world.spawn_actor(walker_obstacle_bp, obstacle_init_trans, attach_to = vehicle)
    actors.append(walker_obstacle_sensor)

    # Ρύθμιση Collision Detector για καταμέτρηση συγκρούσεων και τοποθέτηση στον μπροστινό προφυλακτήρα του οχήματος 
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_init_trans = carla.Transform(carla.Location(x = vehicle.bounding_box.extent.x + 0.1, z = 0.8)) 
    collision_sensor = world.spawn_actor(collision_bp, collision_init_trans, attach_to = vehicle)
    actors.append(collision_sensor)

    # Αρχικοποίηση δεδομένων Semantic LiDAR
    point_list = o3d.geometry.PointCloud()

    # Ανάκτηση διαστάσεων σε pixels και μετατροπή σε ακεραίους
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float() 

    # Παράμετροι κάμερας για την προβολή από 3D σε 2D
    K = build_projection_matrix(image_w, image_h, fov) 

    # Αρχικοποίηση δεδομένων OpenCV παραθύρου
    camera_data = {'rgb_image': np.zeros((image_h, image_w, 4)), 
                   'obstacle': [],
                   'collision': False,
                   'walker_collision_count': 0,
                   'vehicle_collision_count': 0}

    # Λειτουργία των αισθητήρων
    camera.listen(lambda image: rgb_callback(image, camera_data))
    lidar.listen(lambda data: semantic_lidar_callback(data, point_list))
    vehicle_obstacle_sensor.listen(lambda event: vehicle_obstacle_callback(event, camera_data, camera, K))
    walker_obstacle_sensor.listen(lambda event: walker_obstacle_callback(event, camera_data, camera, K))
    collision_sensor.listen(lambda event: collision_callback(event, camera_data))

    # Εύρεση της μεγαλύτερης διαδρομής στον χάρτη από το σημείο τοποθέτησης του οχήματος 
    sys.path.append('C:/CARLA-0.10.0/PythonAPI/carla')
    from agents.navigation.global_route_planner import GlobalRoutePlanner # κάνουμε import την κλάση GlobalRoutePlanner

    sampling_resolution = 1 # 1 μέτρο απόσταση ανάμεσα στα spawn_points
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)

    distance = 0
    for spawn_point in spawn_points:

        # route = [(way_point, road_option)]
        cur_route = grp.trace_route(start_point.location, spawn_point.location) # χρησιμοποιούμε την συνάρτηση trace_route της κλάσης GlobalRoutePlanner
        if len(cur_route) > distance:
            distance = len(cur_route)
            route = cur_route

    # Εκτύπωση της μεγαλύτερης διαδρομής
    for i in range(len(route)):
        world.debug.draw_string(route[i][0].transform.location, '*', draw_shadow = False, color = carla.Color(r=0, g=0, b=255), life_time = 180.0, persistent_lines = True) # route[i][0] είναι το κάθε way_point

    # !!! V2X !!!
    # Εκκίνηση broker 
    print("\n[INFO] Starting broker\n")

    broker_process = subprocess.Popen(
        ["python", "C:/CARLA-0.10.0/PythonAPI/examples/RSU.py"],
        #stdout=subprocess.DEVNULL,   # κανένα print στο cmd
        #stderr=subprocess.DEVNULL    # κανένα error στο cmd
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(1) # για την εκτέλεση του RSU.py

    V2X_clients_process = subprocess.Popen(
        ["python", "C:/CARLA-0.10.0/PythonAPI/examples/V2X_clients.py"],
        #stdout=subprocess.DEVNULL,   # κανένα print στο cmd
        #stderr=subprocess.DEVNULL,   # κανένα error στο cmd
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(1) # για την εκτέλεση του V2X_clients.py

    # Socket για λήψη V2X message από τον broker
    broker_to_ego_socket = socket(AF_INET, SOCK_DGRAM)
    broker_to_ego_socket.bind(("localhost", 5555))
    broker_to_ego_socket.settimeout(0.0001)
    # !!!!!!!!!!!

    # Ρυθμίσεις εικόνας OpenCV παραθύρου (αρχική εμφάνιση εικόνας)
    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB Camera', camera_data['rgb_image'])
    cv2.waitKey(1)

    # Ρυθμίσεις Open3D παραθύρου για την εμφάνιση δεδομένων Semantic LiDAR (αρχική εμφάνιση)
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name = 'Semantic LiDAR',
        width = 960,
        height = 540,
        left = 480,
        top = 270)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    add_open3d_axis(vis)

    # Εύρεση STOP και YIELD signs 
    stops = world.get_actors().filter("traffic.stop") 
    yields = world.get_actors().filter("traffic.yield") # φαίνονται 2 yield signs σαν actors αλλά στην πράξη είναι και τα δύο το ίδιο yield sign

    # Μεταβλητές για STOP και YIELD signs
    stop_timers = {}      # κρατάμε arrival_time για κάθε STOP
    passed_stops = set()  # κρατάμε ids των STOP που έχουμε περάσει
    passed_yields = set() # κρατάμε ids των YIELD που έχουμε περάσει

    # Μεταβλητή για Collision Detector
    collision_counter = 20

    current_index = 0 # το way_point
    vehicle_start_time = time.time() # χρόνος έναρξης οχήματος
    
    while True:

        frame = world.tick()
        
        # !!! V2X !!!
        data = None
        while True: # διαβάζω όλα τα V2X μηνύματα που αντιστοιχούν στο frame μέχρι να αδειάσει ο buffer
            try:
                data = broker_to_ego_socket.recv(4096) # από μόνο του κολλάει μέχρι να λάβει μήνυμα
                total_received += 1
                handle_V2X_messages(data)
            except timeout:
                break  
        # !!!!!!!!!!!
        
        v = vehicle.get_velocity()
        speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2),0) # μετατροπή της ταχύτητας από m/s σε km/h
        image = camera_data['rgb_image']
        image = cv2.putText(image, 'Speed: ' + str(int(speed)) + ' kmh', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # cv2.putText(image, text, font, font_scale, colour, thickness, line_type)

        image = cv2.putText(image, 'Walker collisions: ' + str(camera_data['walker_collision_count']), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 ,255), 1, cv2.LINE_AA)
        image = cv2.putText(image, 'Vehicle collisions: ' + str(camera_data['vehicle_collision_count']), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 ,255), 1, cv2.LINE_AA)
        if camera_data['collision']:
            collision_counter -= 1
            if collision_counter <= 1:
                camera_data['collision'] = False
            image = cv2.putText(image, 'Collision', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, 2)
        else:
            collision_counter = 20

        # Απόσταση μεταξύ οχήματος και waypoint, ελέγχω αν το όχημα βρίσκεται στο τρέχον waypoint και αν υπάρχει επόμενο ενημερώνω την μεταβλητή μου
        if vehicle.get_transform().location.distance(route[current_index][0].transform.location) < 2.0:
            if current_index < len(route) - 1:
                current_index = current_index + 1
            else:
                print("\nArrived at destination")
                vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = 0.0, brake = 1.0))
                break

        # Διαφορά γωνίας μεταξύ επόμενου waypoint και οχήματος (τρέχοντος waypoint) 
        dx = route[current_index][0].transform.location.x - vehicle.get_transform().location.x
        dy = route[current_index][0].transform.location.y - vehicle.get_transform().location.y

        target_yaw = math.atan2(dy, dx)                                     # γωνία μεταξύ δυο waypoints, γωνία που πρέπει να έχει το όχημα 
        vehicle_yaw = math.radians(vehicle.get_transform().rotation.yaw)    # γωνία που έχει το όχημα
        angle_diff = target_yaw - vehicle_yaw                               # πόσο πρέπει να στρίψει το όχημα
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff)) # περιορισμός σε [-180, +180], ώστε να επιλέγει το όχημα την μικρότερη γωνία περιστροφής, πχ αν target_yaw = -170 και vehicle_yaw = 170, τότε angle_diff = -340, επομένως επιλέγει να κάνει περιστροφή +20

        steer_value = angle_diff / math.radians(25) # για το όχημα η μέγιστη γωνία περιστροφής του τροχού προς μια κατεύθυνση είναι 70 μοίρες, όμως πρόβλημα στις στροφές 
        steer_value = max(-1, min(1, steer_value)) 

        # Μεταβλητές για STOP και YIELD signs 
        closest_stop = min(stops, key=lambda s: vehicle.get_transform().location.distance(s.get_transform().location))
        stop_distance = vehicle.get_transform().location.distance(closest_stop.get_transform().location) # εύρεση κοντινότερου STOP από το όχημα μας
        yield_distance = vehicle.get_transform().location.distance(yields[0].get_transform().location)   # εύρεση κοντινότερου YIELD από το όχημα μας

        action_taken = False

        # Προτεραιότητα 1-Ανίχνεση πεζού
        '''
        if walker_detected: # sensors
            action_taken = True
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            walker_detected = False  

        if v2x_flags["walker"]: # V2X 
            action_taken = True
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            
            if v2x_hold["walker"] > 0:
                v2x_hold["walker"] -= 1
            else:
                v2x_flags["walker"] = False
        '''

        if walker_detected or v2x_flags["walker"]: # sensors+V2X 
            action_taken = True
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            
            if walker_detected:
                walker_detected = False
            if v2x_flags['walker']:
                if v2x_hold["walker"] > 0:
                    v2x_hold["walker"] -= 1
                else:
                    if not walker_detected:
                        v2x_flags["walker"] = False
            
        # Προτεραιότητα 2-Ανίχνεση οχήματος
            '''
            elif vehicle_detected: # sensors
                action_taken = True
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                vehicle_detected = False
                
            elif v2x_flags["vehicle"]: # V2X
                action_taken = True
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                
                if v2x_hold["vehicle"] > 0:
                    v2x_hold["vehicle"] -= 1
                else:
                    v2x_flags["vehicle"] = False
            '''
                
        elif vehicle_detected or v2x_flags["vehicle"]: # sensors+V2X 
            action_taken = True
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))

            if vehicle_detected:
                vehicle_detected = False
            if v2x_flags['vehicle']:
                if v2x_hold["vehicle"] > 0:
                    v2x_hold["vehicle"] -= 1
                else:
                    if not vehicle_detected:
                        v2x_flags["vehicle"] = False  
        
        # Προτεραιότητα 3-Ανίχνεση κόκκινου/πορτοκαλί φαναριού
        elif vehicle.get_traffic_light() is not None and vehicle.get_traffic_light_state().name in ["Red", "Yellow"]: # sensors only
            action_taken = True
            vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = 0.0, brake = 0.5))
            print(f"Detected traffic light with id={vehicle.get_traffic_light().id}, state={vehicle.get_traffic_light_state()}")          

        # Προτεραιότητα 4-Ανίχνεση STOP sign 
        elif (stop_distance<6 and closest_stop.id not in passed_stops): 
            action_taken = True 
    
            # Αποθήκευση arrival_time μόνο την πρώτη φορά
            if closest_stop.id not in stop_timers:
                print("\nSTOP sign detected -> braking, starting 2s timer")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                frame = world.tick()                    
                stop_timers[closest_stop.id] = world.get_snapshot().timestamp.elapsed_seconds
                arrival_time = stop_timers[closest_stop.id]

            # Αν έχουν περάσει 2 δευτερόλεπτα
            current_time = world.get_snapshot().timestamp.elapsed_seconds
            if current_time-arrival_time>=2:
                print("\n2s timer has finished -> checking for vehicles")

                # Ελέγχουμε για οχήματα
                lidar_snapshot = lidar_data.copy()
                labels = lidar_snapshot[:, 5].astype(np.uint32)
                mask = np.isin(labels, [14, 15, 16])
                obstacle_points = lidar_snapshot[mask] # Φιλτράρουμε οχήματα
                obstacle_ids = np.unique(obstacle_points[:, 4]) # Παίρνουμε όλα τα μοναδικά obstacle_ids
                obstacle_ids = obstacle_ids[1:-1] # βγάζουμε εκτός το πρώτο obstacle_id το οποίο δεν ξέρουμε σε ποιον actor αντιστοιχεί, όπως και το τελευταίο που αντιστοιχεί στο όχημα μας
                print(obstacle_ids)

                obstacle_ahead = False

                # Φιλτράρουμε όλα τα οχήματα που εμφανίζονται στον Semantic LiDAR
                for obstacle_id in obstacle_ids:

                    # Points του συγκεκριμένου οχήματος
                    points_of_obstacle = obstacle_points[obstacle_points[:, 4] == obstacle_id]

                    # Centroid του οχήματος, πίνακας με [x, y, z], ως προς το σύστημα του Semantic LiDAR
                    centroid = points_of_obstacle[:, :3].mean(axis=0)
                    print(f"Vehicle {world.get_actors().find(int(obstacle_id)).type_id} with ID {int(obstacle_id)} at centroid: {centroid}")

                    # Ελέγχουμε αν υπάρχει όχημα μπροστά μας στην διασταύρωση που να κινείται
                    if 0<centroid[0]<34 and -17<centroid[1]<17:
                        print(f"Vehicle {world.get_actors().find(int(obstacle_id)).type_id} with ID {int(obstacle_id)} at centroid: {centroid} -> stay stopped")
                        obstacle_ahead = True
                        break

                # Αν δεν έχει ανιχνευθεί κανένα όχημα, επιταχύνουμε 
                if obstacle_ahead==False:
                    print("\nPath clear -> proceed\n")
                    vehicle.apply_control(carla.VehicleControl(throttle = maintain_speed(speed), steer = steer_value, brake = 0.0))
                    passed_stops.add(closest_stop.id) 
                    stop_timers.pop(closest_stop.id, None)  
      
        # Προτεραιότητα 5-Ανίχνεση YIELD sign
        elif (yield_distance<6 and yields[0].id not in passed_yields):
            action_taken = True
    
            # Φτάνουμε στο YIELD
            print("\nYIELD sign detected")

            frame = world.tick()

            # Ελέγχουμε για οχήματα
            lidar_snapshot = lidar_data.copy()
            labels = lidar_snapshot[:, 5].astype(np.uint32)
            mask = np.isin(labels, [14, 15, 16])
            obstacle_points = lidar_snapshot[mask] # Φιλτράρουμε οχήματα
            obstacle_ids = np.unique(obstacle_points[:, 4]) # Παίρνουμε όλα τα μοναδικά obstacle_ids
            obstacle_ids = obstacle_ids[1:-1] # βγάζουμε εκτός το πρώτο obstacle_id το οποίο δεν ξέρουμε σε ποιον actor αντιστοιχεί, όπως και το τελευταίο που αντιστοιχεί στο όχημα μας
            print(obstacle_ids)

            obstacle_ahead = False

            # Φιλτράρουμε όλα τα οχήματα που εμφανίζονται στον Semantic LiDAR
            for obstacle_id in obstacle_ids:

                # Points του συγκεκριμένου οχήματος
                points_of_obstacle = obstacle_points[obstacle_points[:, 4] == obstacle_id]

                # Centroid του οχήματος, πίνακας με [x, y, z], ως προς το σύστημα του Semantic LiDAR
                centroid = points_of_obstacle[:, :3].mean(axis=0)
                print(f"Vehicle {world.get_actors().find(int(obstacle_id)).type_id} with ID {int(obstacle_id)} at centroid: {centroid}")

                # Ελέγχουμε αν υπάρχει όχημα μπροστά μας στην διασταύρωση που να κινείται
                if 0<centroid[0]<12 and -7<centroid[1]<15: # centroid[1] ανάποδα από άξονες οχήματος (lidar_data -> παίρνουμε τα data του point cloud όπου το y δεν έχει αλλάξει ακόμα πρόσημο)
                    print(f"Vehicle {world.get_actors().find(int(obstacle_id)).type_id} with ID {int(obstacle_id)} at centroid: {centroid} -> stay stopped")
                    vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = 0.0, brake = 1.0))
                    obstacle_ahead = True
                    break
            
            # Αν δεν έχει ανιχνευθεί κανένα όχημα συνεχίζουμε
            if obstacle_ahead==False:
                print("\nPath clear -> proceed\n")
                vehicle.apply_control(carla.VehicleControl(throttle = maintain_speed(speed), steer = steer_value))
                passed_yields.add(yields[0].id)

        if not action_taken:
            vehicle.apply_control(carla.VehicleControl(throttle = maintain_speed(speed), steer = steer_value))
            
        # Συνθήκη ώστε το όχημα να σταματήσει αν περάσει ξανά από το ίδιο STOP 
        if stop_distance>20 and closest_stop.id in passed_stops:
            passed_stops.remove(closest_stop.id)

        # Συνθήκη ώστε το όχημα να σταματήσει αν περάσει ξανά από το ίδιο YIELD 
        if yield_distance>20 and yields[0].id in passed_yields:
            passed_yields.remove(yields[0].id)

        # Απεικόνιση των δεδομένων LiDAR στο παράθυρο Open3D και διαρκής ανανέωση τους 
        if not lidar_added and len(point_list.points)>0:
            vis.add_geometry(point_list)
            lidar_added = True

        if lidar_added:
            vis.update_geometry(point_list)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.005)

        # Απεικόνιση της εικόνας του OpenCV παραθύρου
        cv2.imshow('RGB Camera', image)

        if cv2.waitKey(1)==ord('q'):
            break

finally:
    print(f"Arrival time: {time.time()-vehicle_start_time} seconds")

    # !!! V2X !!!
    print(f"V2X Messages received: {total_received}")
    if len(latencies)>0:
        print(f"Average V2X Latency: {sum(latencies)/len(latencies)}")
    # !!!!!!!!!!!

    print("Destroying actors")
    for actor in actors:
        actor.destroy()
    
    print("Destroying windows")
    cv2.destroyAllWindows()
    vis.destroy_window()

    # !!! V2X !!!
    broker_process.terminate()
    # !!!!!!!!!!!