# Adding new YCB objects to pandapybullet

1.  Create a folder with the name of your object (we will use "hammer" for this example).

2.  Go to the YCB model set website [here](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/).

3.  Download the 16k laser scan files associated to your object.

4.  Extract the following files from the archive and put them into the "hammer" folder : `textured.mtl`, `textured.obj`, `textured.dae` and `texture_map.png`.

5.  Create in the "hammer" folder an URDF named "hammer.urdf" with the following structure:
```xml
<?xml version="1.0" ?>
<robot name="hammer.urdf">
  <link name="base_link">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
	    <mesh filename="textured.obj" scale="1 1 1"/>
      </geometry>
        <material name="white">
          <color rgba="1 1 1 1"/>
        </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
	    <mesh filename="textured.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
```

6.  Use the loadURDF function of pybullet to import your model.
