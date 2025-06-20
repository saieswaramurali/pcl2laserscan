import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class DepthToLaserScanNode(Node):
    def __init__(self):
        super().__init__('depth_to_laserscan')
        # RealSense D455 intrinsics (change if your camera differs)
        self.fx = 617.0
        self.fy = 617.0
        self.cx = 320.0
        self.cy = 240.0

        self.angle_min = -1.57  # -90 degrees
        self.angle_max = 1.57   # +90 degrees
        self.angle_increment = (self.angle_max - self.angle_min) / 640.0  # width of image

        self.range_min = 0.3
        self.range_max = 5.0

        self.depth_topic = '/camera/camera/depth/image_rect_raw'
        self.scan_topic = '/camera/scan'

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.pub = self.create_publisher(LaserScan, self.scan_topic, 10)

    def depth_callback(self, msg):
        try:
            # Most RealSense use 16UC1 (millimeters)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        if len(cv_image.shape) != 2:
            self.get_logger().error('Depth image is not 2D!')
            return

        # Take the center row of the depth image
        center_row = cv_image[cv_image.shape[0] // 2, :].astype(np.float32) / 1000.0  # convert mm to meters
        # Mask out invalid values
        valid = np.logical_and(center_row > self.range_min, center_row < self.range_max)
        ranges = np.where(valid, center_row, float('inf'))

        scan = LaserScan()
        scan.header = msg.header
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges.tolist()
        self.pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToLaserScanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()