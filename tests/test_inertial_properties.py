import xml.etree.ElementTree as ET

def test_mjcf_inertias_positive():
    root=ET.parse('description/mjcf/two_wheel_robot.xml').getroot()
    for inertial in root.findall('.//inertial'):
        assert all(float(x)>0 for x in inertial.attrib['diaginertia'].split())
