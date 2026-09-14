import importlib.util
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "scripts", "prepare_contact_model.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_contact_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContactModelTests(unittest.TestCase):
    def test_injects_bumper_into_base_collision(self):
        source = """<?xml version='1.0'?>
<sdf version='1.6'><model name='burger'>
<link name='wheel'><collision name='wheel_collision'/></link>
<link name='base_link'><collision name='base_collision'/></link>
</model></sdf>"""
        directory = tempfile.gettempdir()
        pid = os.getpid()
        input_path = os.path.join(directory, f"_contact_input_{pid}.sdf")
        output_path = os.path.join(directory, f"_contact_output_{pid}.sdf")
        try:
            with open(input_path, "w", encoding="utf-8") as stream:
                stream.write(source)
            MODULE.inject_contact_sensor(input_path, output_path)
            root = ET.parse(output_path).getroot()
        finally:
            for path in (input_path, output_path):
                if os.path.exists(path):
                    os.remove(path)
        sensor = root.find("./model/link[@name='base_link']/sensor")
        self.assertIsNotNone(sensor)
        self.assertEqual(sensor.findtext("contact/collision"), "base_collision")
        self.assertEqual(sensor.find("plugin").get("filename"), "libgazebo_ros_bumper.so")
        self.assertEqual(sensor.findtext("plugin/ros/remapping"), "bumper_states:=bumper_states")

    def test_rate_locks_lidar_and_can_require_its_presence(self):
        source = """<sdf version='1.6'><model name='burger'><link name='base_link'>
<collision name='base_collision'/><sensor name='lds' type='ray'><update_rate>5</update_rate></sensor>
</link></model></sdf>"""
        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, "in.sdf")
            output_path = os.path.join(tmp, "out.sdf")
            with open(input_path, "w", encoding="utf-8") as stream:
                stream.write(source)
            MODULE.inject_contact_sensor(input_path, output_path, lidar_update_rate=20.0, require_lidar=True)
            root = ET.parse(output_path).getroot()
            self.assertEqual(root.findtext(".//sensor[@name='lds']/update_rate"), "20.0")

            without_lidar = os.path.join(tmp, "without.sdf")
            with open(without_lidar, "w", encoding="utf-8") as stream:
                stream.write("<sdf version='1.6'><model name='burger'><link name='base_link'><collision name='c'/></link></model></sdf>")
            with self.assertRaisesRegex(ValueError, "no ray LiDAR"):
                MODULE.inject_contact_sensor(without_lidar, output_path, require_lidar=True)


if __name__ == "__main__":
    unittest.main()
