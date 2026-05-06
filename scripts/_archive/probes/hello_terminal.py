import time, os, sys
print(f"HELLO from pid={os.getpid()} session can be seen!")
print(f"Python: {sys.executable}")
for i in range(60):
    print(f"tick {i}")
    time.sleep(1)
