import serial
import csv
import time

glove = serial.Serial("/dev/ttyUSB0", baudrate=115200)

log_file = open("quat.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(
    ["step", "time", "w", "x", "y", "z"]
)
step = 0
t = time.time()
while 1:
    step += 1
    a = glove.readline().decode("utf-8")
    try: 
        digits = list(map(float, a[1:-4].split(',')))
        print(digits)
        print(len(digits))

        log_writer.writerow([step, time.time() - t, digits[1], digits[2], digits[3], digits[4]])
    except:
        print("SOS")
    