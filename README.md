<h1>INTELLIGENT AGENT FOR THE VISION IMPAIRED</h1>
<h3>By: Jordan Arbeeny and Braden Harris 
<h4>Christian Brothers Academy</h4>
<h4>TSA 2026</h4>

<h4>WHAT IS IT?</h4> A wearable AI powered device designed to help blind and visually impaired people navigate the world safely. The program uses a camera to detect objects and people in real time and tells the user what is around them through voice alerts.

<h4>HOW IT WORKS</h4>
<ol>
  <li>
Object Detection
Uses a YOLOv8 AI model to detect over 80 different objects
Draws colored boxes around everything it sees
Tells the user what the object is and where it is
  </li>
<li>
Distance Warning
Measures how close objects are based on box size
Green = far away, safe
Orange = getting close, caution
Red = very close, danger
  </li>
<li>
Position Alerts
Tells the user if something is on their left, right, or straight ahead
Example: "Warning! Chair on your left"
  </li>
<li>
Face Recognition
Detects and recognizes familiar faces
Learns and remembers faces permanently
Says the persons name when it sees them
Example: "Warning! Mom on your right"
  </li>
<li>
Voice Alerts
Speaks all warnings out loud in real time
3 different voices to choose from
Can be muted and unmuted at any time
  </li>
<li>
Time Awareness
Announces the time when the program starts
Says good morning, good afternoon, or good evening
Updates the time every 10 minutes
  </li>
<li>
Clear Path
Says "Clear path ahead" when nothing is detected nearby
</li>
</ol>
<br>

<h4>KEYBOARD COMMANDS</h4>
<table>
<tr><th>Key</th><th>Action</th></tr>
<tr><td>P</td><td>Mute / Unmute Voice</td></tr>
<tr><td>V</td><td>Cycle through voices</td></tr>
<tr><td>N</td><td>Name a detected face</td></tr>
<tr><td>Q</td><td>Quit</td></tr>

  
</table>


<h4>TECHNOLOGY USED</h4>
Python 3.11
OpenCV 
YOLOv8
Windows Speech Synthesis
NumPy

<h4>INSTALLATION INSTRUCTIONS</h4>

STEP 1 — Install Python
Go to https://www.python.org/downloads/release/python-31115/
Download Python 3.11.x
Run installer and check "Add Python to PATH"
Click Install Now

STEP 2 — Open Terminal
Press Windows + R
Type <code>powershell</code>
Hit Enter


STEP 3 — Install Libraries Copy and paste each line and hit Enter:
<code>
python -m pip install opencv-python
python -m pip install opencv-contrib-python
python -m pip install ultralytics
python -m pip install numpy
python -m pip install pyttsx3
</code>

STEP 4 - Copy the file TSA_project.py from this repository to a folder on your computer

STEP 5 — Change to the download folder and run the Program
<code>
cd &lt;TSA Project Folder&gt;
python TSA_project.py
</code>


<h4>Screenshots</h4>

<img width="948" height="529" alt="Screenshot 2026-03-31 204311" src="https://github.com/user-attachments/assets/f19a899c-ce0c-4948-8eac-bd11702a5b37" />

This screenshot shows a known face recognized by the application. Once recognized, the application will speak the name of the person that was detected.

<img width="950" height="713" alt="Screenshot 2026-03-31 203618" src="https://github.com/user-attachments/assets/c75cefc0-c4dc-4643-aaca-2dd427d2d66a" />

This screenshot shows the welcome banner of the application. It is purposely displayed in high contract large fonts for the visually impaired.





