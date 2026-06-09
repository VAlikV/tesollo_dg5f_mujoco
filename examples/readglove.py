import serial


glove = serial.Serial("/dev/ttyUSB0", baudrate=115200)

while 1:
    a = glove.readline().decode("utf-8")
    try: 
        digits = list(map(float, a[1:-4].split(',')))
        print(digits)
        print(len(digits))
    except:
        print("SOS")
    