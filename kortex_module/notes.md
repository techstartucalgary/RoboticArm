# General Architecture

World (env) -> RPi -> AWS -> Python -> Robot  
  ^                                      |
  |_______________________________________

# Python script flow
Actions:
- each motor has value in a given range
  - these values can be scaled as necessary, will likely be [-360,360] or [-180,180] where the value represents the position of the actuator
- each step will send value for each actuator within this range and the arm should move the actuators to that position
- will also receive x,y,z coordinates for gripper (do later)
- Example: [ actuator1_val, actuator2_val, actuator3_val, actuator4_val, actuator5_val, actuator6_val, actuator7_val, gripper_actuator (between 0 - 255)]