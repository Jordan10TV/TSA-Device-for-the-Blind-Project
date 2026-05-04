REQUIREMENTS
Windows 10 or 11
A webcam (built in or USB)
Internet connection (first run only)
Python 3.11.9

STEP 1 — Install Python
Go to python.org/downloads
Download Python 3.11.9
Run the installer
Check the box that says "Add Python to PATH" before clicking install
Click Install Now

STEP 2 — Download the project files Download the zip file we provided and extract it to your Downloads folder

STEP 3 — Open terminal
Press Windows key + R
Type powershell and hit Enter

STEP 4 — Install required libraries Copy and paste these one by one and hit Enter after each:
python -m pip install opencv-python
python -m pip install opencv-contrib-python
python -m pip install ultralytics
python -m pip install numpy
python -m pip install pyttsx3

STEP 5 — Run the program
cd Downloads
python TSA_project.py

CONTROLS
Key
Action
P
Mute / Unmute voice
V
Cycle through voices
N
Name a detected face
Q
Quit the program


HOW TO ADD A FACE
Stand in front of camera
Press N
Type the person's name in the terminal and hit Enter
Slowly move your head left, right, up and down
Wait for "Got it! I will remember [name]"

HOW TO RESET SAVED FACES
Run these in the terminal:
del face_database.json
del face_samples.pkl
del face_model.yml

