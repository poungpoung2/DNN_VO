
import rosbag
from pathlib import Path
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import csv
import cv2

DATA_DIR = Path("data")

class Pose:
    def __init__(self, msg):
        self.x, self.y, self.z = msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z
        self.to_euler(msg.transform.rotation)

    def to_euler(self, q):
        yaw = np.arctan2(2.0 * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
        pitch = np.arcsin(np.clip(2.0 * (q.w * q.y - q.x * q.z), -1.0, 1.0))
        roll = np.arctan2(2.0 * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        
    def return_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll
        }
        

        
def process_bag_file(file_path):
    bag_path = Path(file_path)

    bridge = CvBridge()
    bag = rosbag.Bag(bag_path)

    image_timestamps = []

    images_cam0 = {}
    images_cam1 = {}

    poses = {}       
    pose_timestamps = []

    for topic, msg, t in bag.read_messages(topics=['/cam0/image_raw', '/cam1/image_raw', '/vicon/firefly_sbx/firefly_sbx', 'vicon/firefly_sbx/firefly_sbx']):
        timestamp = t.to_sec()

        if topic == "/cam0/image_raw":
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            images_cam0[timestamp] = cv_image
            image_timestamps.append(timestamp)
            
        if topic == "/cam1/image_raw":
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            images_cam1[timestamp] = cv_image

        if topic == "vicon/firefly_sbx/firefly_sbx" or topic == "/vicon/firefly_sbx/firefly_sbx":
            pose = Pose(msg)
            poses[timestamp] = pose
            pose_timestamps.append(timestamp)


    print(f"Image: {len(image_timestamps)} | Pose: {len(pose_timestamps)}")

    image_pose_pairs = []
    max_time_diff = 0.01
    
    pose_timestamps_copy = pose_timestamps.copy()  

    for img_time in image_timestamps:
        if not pose_timestamps_copy: 
            break
        
        closest_time = min(pose_timestamps_copy, key=lambda x: abs(x - img_time))
        time_diff = abs(closest_time - img_time)
                
        # Only use pairs within the maximum time difference
        if time_diff <= max_time_diff:
            image_pose_pairs.append((img_time, closest_time))
        
        # Avoid duplication
        pose_timestamps_copy.remove(closest_time)
        
    
    file_name = file_path.stem
    
    top_dir = DATA_DIR / file_name
    
    cam0_dir = top_dir / "cam0"
    cam0_dir.mkdir(exist_ok = True, parents = True)
    
    cam1_dir = top_dir / "cam1"
    cam1_dir.mkdir(exist_ok = True, parents = True)
    
    csv_path = top_dir / "poses.csv"

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        for i, (img_time, pose_time) in enumerate(image_pose_pairs):
            if img_time in images_cam0:
                cam0_img = images_cam0[img_time]
                cam0_path = cam0_dir / f"{i:06d}.jpg"
                cv2.imwrite(str(cam0_path), cam0_img)
            
            if img_time in images_cam1:
                cam1_img = images_cam1[img_time]
                cam1_path = cam1_dir / f"{i:06d}.jpg"
                cv2.imwrite(str(cam1_path), cam1_img)

            pose = poses[pose_time]
            pose_dict = pose.return_dict()
            
            writer.writerow({
                'image': f"{i}",  
                'x': f"{pose_dict['x']:.6f}",
                'y': f"{pose_dict['y']:.6f}",
                'z': f"{pose_dict['z']:.6f}",
                'roll': f"{pose_dict['roll']:.6f}",
                'pitch': f"{pose_dict['pitch']:.6f}",
                'yaw': f"{pose_dict['yaw']:.6f}"
            })

    print(f"Processed {len(image_pose_pairs)} image-pose pairs from {file_path}")



def compile_all_rosbags(dir_path):
    rosbags_files = list(dir_path.glob("*.bag"))
    
    for bag_file in rosbags_files:
        process_bag_file(bag_file)
        
    print(f"Done processing {len(rosbags_files)} ros bags")
    

if __name__ == "__main__":
    ros_path = Path("rosbags")
    compile_all_rosbags(ros_path)

