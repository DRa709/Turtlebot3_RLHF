from setuptools import setup, find_packages

setup(
    name="turtlebot3_drl_nav",
    version="2.0.0",
    description="Deep Reinforcement Learning & RLHF Navigation Suite for TurtleBot3",
    author="Dhruv Shankar Ray, Asha Barua",
    author_email="dhruvshankar@vt.edu, ashabarua@vt.edu",
    maintainer="Dhruv Shankar Ray",
    maintainer_email="dhruvshankar@vt.edu",
    url="https://github.com/DRa709/Turtlebot3_RLHF",
    packages=find_packages(include=["turtlebot3_drl_nav", "turtlebot3_drl_nav.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pyyaml>=5.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Robotics",
    ],
)
