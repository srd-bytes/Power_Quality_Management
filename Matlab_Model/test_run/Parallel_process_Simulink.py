import socket, struct, numpy as np
from multiprocessing import Process, Event
from multiprocessing.shared_memory import SharedMemory
import time

# UDP_IP = "127.0.0.1"
# UDP_PORT = 6001
# SAMPLE_COUNT = 1000
# FEATURE_COUNT = 37   # sample_idx + 36 doubles
# SHAPE = (SAMPLE_COUNT, FEATURE_COUNT)


# def logger_process(shm_name_A, shm_name_B, ready_A, ready_B):
#     shmA = SharedMemory(name=shm_name_A)
#     shmB = SharedMemory(name=shm_name_B)

#     bufA = np.ndarray(SHAPE, dtype=np.float64, buffer=shmA.buf)
#     bufB = np.ndarray(SHAPE, dtype=np.float64, buffer=shmB.buf)

#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.bind((UDP_IP, UDP_PORT))

#     print("[Logger] Running...")

#     batch_idx = 0
#     sample_idx = 0

#     while True:
#         # Decide which buffer to fill
#         target_buf = bufA if batch_idx % 2 == 0 else bufB
#         target_flag = ready_A if batch_idx % 2 == 0 else ready_B

#         for i in range(SAMPLE_COUNT):
#             data, _ = sock.recvfrom(4096)
#             values = struct.unpack("d"*36, data)
#             row = [sample_idx] + list(values)
#             target_buf[i, :] = row
#             sample_idx += 1

#         # Mark buffer as ready
#         target_flag.set()
#         batch_idx += 1




import pandas as pd


UDP_IP = "127.0.0.1"
UDP_PORT = 25000

SAMPLE_COUNT = 20
FEATURE_COUNT = 85  # sample_idx + 84 float values (14 buses × 6 values)
SHAPE = (SAMPLE_COUNT, FEATURE_COUNT)

# ----------------------------
# Logger Process Function
# ----------------------------
def logger_process(shm_name_A, shm_name_B, ready_A: Event, ready_B: Event):

    # Attach shared memory
    shmA = SharedMemory(name=shm_name_A)
    shmB = SharedMemory(name=shm_name_B)

    bufA = np.ndarray(SHAPE, dtype=np.float64, buffer=shmA.buf)
    bufB = np.ndarray(SHAPE, dtype=np.float64, buffer=shmB.buf)

    # UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print("[Logger] Listening on UDP {}:{}".format(UDP_IP, UDP_PORT))

    batch_idx = 0
    global_sample_index = 0

    # struct format for 84 float32 values
    UNPACK_FORMAT = "<84f"      # use "<84d" if Simulink sends double
    PACKET_SIZE = 84 * 4        # update to 8 if using double

    while True:
        # Select buffer A or B
        target_buf = bufA if batch_idx % 2 == 0 else bufB
        target_flag = ready_A if batch_idx % 2 == 0 else ready_B

        print(f"[Logger] Filling batch {batch_idx}")

        for i in range(SAMPLE_COUNT):
            packet, _ = sock.recvfrom(4096)
            if len(packet) != PACKET_SIZE:
                print("⚠ Warning: Wrong packet size:", len(packet))
                continue

            # unpack 84 floats
            values = struct.unpack(UNPACK_FORMAT, packet)  # tuple length 84

            # Build row: [sample_index, 84 float values]
            row = np.zeros(FEATURE_COUNT)
            row[0] = global_sample_index
            row[1:] = values

            target_buf[i, :] = row
            global_sample_index += 1

        # Mark buffer ready
        target_flag.set()

        # ----------------------------
        # SAVE BATCH TO EXCEL
        # ----------------------------
        timestamp = int(time.time())
        filename = f"bus_data_batch_{batch_idx}_{timestamp}.xlsx"

        # Convert to DataFrame with labeled columns
        columns = ["sample_index"]

        # Create labels like Bus1_Va, Bus1_Vb, Bus1_Vc, Bus1_Ia, Bus1_Ib, Bus1_Ic, etc.
        for bus in range(1, 15):          # 14 buses
            for name in ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]:
                columns.append(f"Bus{bus}_{name}")

        df = pd.DataFrame(target_buf, columns=columns)
        df.to_excel(filename, index=False)

        print(f"[Logger] Saved batch {batch_idx} to Excel: {filename}")

        batch_idx += 1



def analyzer_process(shm_name_A, shm_name_B, ready_A, ready_B):
    shmA = SharedMemory(name=shm_name_A)
    shmB = SharedMemory(name=shm_name_B)

    bufA = np.ndarray(SHAPE, dtype=np.float64, buffer=shmA.buf)
    bufB = np.ndarray(SHAPE, dtype=np.float64, buffer=shmB.buf)

    print("[Analyzer] Running...")

    batch_idx = 0

    while True:
        # Choose buffer to read
        source_buf = bufA if batch_idx % 2 == 0 else bufB
        source_flag = ready_A if batch_idx % 2 == 0 else ready_B

        # Wait for logger to fill buffer
        source_flag.wait()

        # PROCESS THE DATA HERE
        arr = source_buf  # No copy – this is shared memory!

        # avg_v = arr[500, 9]  # example: voltage1_phA index
        # print(f"[Analyzer] Batch {batch_idx} avg V1A = {avg_v:.2f}")
        print(f"[Analyzer] Batch {batch_idx} avg V1A = {np.shape(arr)}")

        # Clear flag
        source_flag.clear()

        batch_idx += 1



if __name__ == "__main__":
    # Create two shared memory blocks
    shmA = SharedMemory(create=True, size=np.zeros(SHAPE).nbytes)
    shmB = SharedMemory(create=True, size=np.zeros(SHAPE).nbytes)

    ready_A = Event()
    ready_B = Event()

    p1 = Process(target=logger_process, args=(shmA.name, shmB.name, ready_A, ready_B))
    p2 = Process(target=analyzer_process, args=(shmA.name, shmB.name, ready_A, ready_B))

    p1.start()
    p2.start()
