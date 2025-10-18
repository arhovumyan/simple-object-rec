from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ros2_object_detection'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='ROS2 Object Detection and Classification Pipeline with YOLO and MobileNet',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = ros2_object_detection.object_detection_node:main',
            'camera_publisher = ros2_object_detection.camera_publisher:main',
        ],
    },
)
