# MovingObjectDetection_MultiCameraSetting


This motion detection application was implemented for the detection of dairy cows use in a loose-housing
barn surveilled by eight cameras, but is also applicable for other multi-camera settings to find moving
objects. USE ON OWN RISK.

• According to the desired application and the analysis in question, the user could define segments
within the fields of view of all cameras used in the installation.
It is recommended to define outer and inner rectangular areas and to use only the inner areas in
the detection of moving objects, to make it less likely that the same object was detected in two
neighbouring rectangles.
In the application for which this tool was designed, different kinds of segments were defined to
split the area of observation: Each lying cubicle was set as a separate segment. Feeding and water
troughs were given a part of the running area as segments in which the cow stands while feeding
or drinking. The running area was cut into larger rectangles. Find the definitions of the resulting
outer (green) and inner (pink) areas in the file ’used_parameter_setting.py’. As could be seen in
the images, the predefined segments do not cover the complete field of view. Areas of no interest
will be neglected by the program.

• Foreground masks were intersected with the inner regions of the predefined segments to specify in
which segment moving objects occur. The minimal size (in pixel) of connected components of the
foreground needs to be set by the user as the parameter px for every camera individually.

• All videos that ought to be processed need to be located in one directory.
The tool operates on a list in which the name of the mp4-file, the px value to use, and the defined
rectangular segments in the field of view of the respective camera were bundled together for each
video to process (see ’used_parameter_setting.py’).
