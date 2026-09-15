"""Inject a ROS2 Gazebo bumper sensor into an installed Burger model SDF."""

import argparse
import os
import xml.etree.ElementTree as ET


def inject_contact_sensor(input_path: str, output_path: str, lidar_update_rate: float = 100.0,
                          require_lidar: bool = False) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()
    model = root.find("model")
    if model is None:
        raise ValueError(f"No <model> in {input_path}")
    lidar_sensors = [
        sensor for sensor in model.findall(".//sensor")
        if sensor.get("type") in ("ray", "gpu_ray")
    ]
    if require_lidar and not lidar_sensors:
        raise ValueError("Burger SDF has no ray LiDAR sensor to rate-lock")
    for lidar in lidar_sensors:
        update = lidar.find("update_rate")
        if update is None:
            update = ET.SubElement(lidar, "update_rate")
        update.text = repr(float(lidar_update_rate))
    diff_drive = model.find(".//plugin[@name='turtlebot3_diff_drive']")
    if diff_drive is not None:
        update = diff_drive.find("update_rate")
        if update is None:
            update = ET.SubElement(diff_drive, "update_rate")
        update.text = "100"
        accel = diff_drive.find("max_wheel_acceleration")
        if accel is None:
            accel = ET.SubElement(diff_drive, "max_wheel_acceleration")
        accel.text = "100.0"
    if model.find(".//sensor[@name='phase1_contact_sensor']") is not None:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return

    candidates = list(model.findall("link"))
    candidates.sort(key=lambda link: 0 if link.get("name") == "base_link" else 1)
    target_link = None
    collision_name = None
    for link in candidates:
        collision = link.find("collision")
        if collision is not None and collision.get("name"):
            target_link = link
            collision_name = collision.get("name")
            break
    if target_link is None or collision_name is None:
        raise ValueError("Burger SDF has no link with a named collision")

    sensor = ET.SubElement(
        target_link,
        "sensor",
        {"name": "phase1_contact_sensor", "type": "contact"},
    )
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = "50.0"
    contact = ET.SubElement(sensor, "contact")
    ET.SubElement(contact, "collision").text = collision_name
    plugin = ET.SubElement(
        sensor,
        "plugin",
        {"name": "phase1_ros_bumper", "filename": "libgazebo_ros_bumper.so"},
    )
    ros = ET.SubElement(plugin, "ros")
    ET.SubElement(ros, "namespace").text = "/"
    ET.SubElement(ros, "remapping").text = "bumper_states:=bumper_states"
    ET.SubElement(plugin, "frame_name").text = target_link.get("name", "base_link")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_sdf")
    parser.add_argument("output_sdf")
    parser.add_argument("--lidar-update-rate", type=float, default=100.0)
    parser.add_argument("--require-lidar", action="store_true")
    args = parser.parse_args()
    inject_contact_sensor(args.input_sdf, args.output_sdf, args.lidar_update_rate, args.require_lidar)
    print(args.output_sdf)


if __name__ == "__main__":
    main()
