import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.cm as cm
from utils import pose_to_trans, rot_to_euler

def visualize_trajectories_with_frames(csv_files, start_pose_idx=0, end_pose_idx=None, axis_length=0.1):
    """
    Visualize multiple trajectories in the same window and draw a coordinate frame 
    (red: x-axis, green: y-axis, blue: z-axis) at each pose.

    Each CSV file should have the following columns:
        - x, y, z: Position coordinates.
        - roll, pitch, yaw: Orientation in radians.
        
    Args:
        csv_files (list of str): List of CSV file paths.
        start_pose_idx (int): Starting index of the pose to visualize.
        end_pose_idx (int, optional): Ending index (exclusive) of the poses to visualize.
                                      If None, all poses from start_pose_idx to the end are plotted.
        axis_length (float): Length of the axis arrows for the coordinate frames.
    """
    
    fig = plt.figure(figsize=(14, 8))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    
    # Generate distinct colors for each trajectory.
    colors = cm.rainbow(np.linspace(0, 1, len(csv_files)))
    
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        # Slice the dataframe based on the starting and ending indices.
        if end_pose_idx is not None:
            df = df.iloc[start_pose_idx:end_pose_idx]
        else:
            df = df.iloc[start_pose_idx:]
        
        # Extract pose data.
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        roll = df['roll'].values
        pitch = df['pitch'].values
        yaw = df['yaw'].values
        
        label = f"Trajectory {i+1}"
        color = colors[i]
        
        # Plot the trajectory line.
        ax3d.plot(x, y, z, marker='o', linestyle='-', color=color, label=label, linewidth=5)
        
        # For each pose, draw the coordinate frame.
        for j in range(len(df)):
            # Create a pose dictionary.
            pose = {
                "position": np.array([x[j], y[j], z[j]]),
                "orientation": np.array([roll[j], pitch[j], yaw[j]])
            }
            # Convert pose to transformation matrix. This function should return a 4x4 matrix.
            T = pose_to_trans(pose)
            pos = T[:3, 3]
            R = T[:3, :3]
            
            # Define local axes in the pose's coordinate frame.
            x_axis = R @ np.array([1, 0, 0])
            y_axis = R @ np.array([0, 1, 0])
            z_axis = R @ np.array([0, 0, 1])
            

    
    ax3d.set_title("3D Trajectories with Local Coordinate Frames")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend()
    
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    csv_files = [
        "/home/undergrad3203/Downloads/data/V1_01_easy/poses.csv", 
        "/home/undergrad3203/Downloads/data/V1_01_easy/deepvo.csv"
    ]
    # Visualize poses from index 100 to 600 (600 is exclusive)
    visualize_trajectories_with_frames(csv_files, start_pose_idx=0, end_pose_idx=200, axis_length=0.1)
