import cv2 # pip install opencv-python
import time
import socket
import threading

# open sockets to send/receive commands/stream to/from drone
host = ''
port = 9000
locaddr = (host,port) 
tello_address = ('192.168.10.1', 8889)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(locaddr)

# function to issue command to drone
# for full list of commands see: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
def command(msg, tello_address):
    msg = msg.encode(encoding="utf-8")
    return sock.sendto(msg, tello_address)

# function that runs on another thread to constantly receive messages being sent from drone
def recv():
    while True: 
        data, server = sock.recvfrom(1518)

# create thread for receving messages
recvThread = threading.Thread(target=recv)
recvThread.daemon = True
recvThread.start()

# establish link with drone
command('command', tello_address)
time.sleep(1)

# turn on video stream
command('streamon', tello_address)
time.sleep(1)

# use opencv to read live video from drone
camera = cv2.VideoCapture('udp://127.0.0.1:11111')
time.sleep(3)

# loop to read and display video
# WARNING: make sure to press q to quit, so properly shuts down
while(True):
    ret, frame = camera.read()
    cv2.imshow('Tello', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# cleanup
sock.close()
camera.release()
cv2.destroyAllWindows()
