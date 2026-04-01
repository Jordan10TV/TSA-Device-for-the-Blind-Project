<h1>BLIND ASSIST GOGGLES</h1>
<h3>By: Jordan Arbeeny and Braden Harris TSA 2026</h3>

<h4>WHAT IS IT?</h4> A wearable AI powered device designed to help blind and visually impaired people navigate the world safely. The program uses a camera to detect objects and people in real time and tells the user what is around them through voice alerts.

<h4>HOW IT WORKS</h4>
1. Object Detection
Uses a YOLOv8 AI model to detect over 80 different objects
Draws colored boxes around everything it sees
Tells the user what the object is and where it is
2. Distance Warning
Measures how close objects are based on box size
Green = far away, safe
Orange = getting close, caution
Red = very close, danger
3. Position Alerts
Tells the user if something is on their left, right, or straight ahead
Example: "Warning! Chair on your left"
4. Face Recognition
Detects and recognizes familiar faces
Learns and remembers faces permanently
Says the persons name when it sees them
Example: "Warning! Mom on your right"
5. Voice Alerts
Speaks all warnings out loud in real time
3 different voices to choose from
Can be muted and unmuted at any time
6. Time Awareness
Announces the time when the program starts
Says good morning, good afternoon, or good evening
Updates the time every 10 minutes
7. Clear Path
Says "Clear path ahead" when nothing is detected nearby

<h4>KEYBOARD COMMANDS</h4>
<table>
<tr><th>Key</th><th>Action</th></tr>
<tr><td>P</td><td>Mute / Unmute Voice</td></tr>
<tr><td>V</td><td>Cycle through voices</td></tr>
<tr><td>N</td><td>Name a detected face/td></tr>
<tr><td>Q</td><td>Quit/td></tr>

  
</table>


<h4>TECHNOLOGY USED</h4>
Python 3.11
OpenCV 
YOLOv8
Windows Speech Synthesis
NumPy

