import cv2
from ultralytics import YOLO
from tkinter import Tk, Label, Entry, Button, Frame, filedialog, IntVar
from tkinter import ttk


from PIL import Image, ImageTk
import threading

# Initialize YOLO model
yolo = YOLO('yolov8s.pt')

# Function to assign unique colors to each class
def get_colours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Global variables
stop_video = False
videoCap = None
target_class = ""
video_path = ""

# Function to select video file
def select_video():
    global video_path
    video_path = filedialog.askopenfilename(
        title="Choose a Video File", 
        filetypes=(("MP4 videos", "*.mp4"), ("All files", "*.*"))
    )
    if video_path:
        video_label.config(text=f"Video Selected: {video_path.split('/')[-1]}")

# Function to start video processing
def start_detection():
    global videoCap, stop_video, target_class, video_path
    if not video_path:
        video_label.config(text="Please pick a video first.")
        return
    stop_video = False
    videoCap = cv2.VideoCapture(video_path)
    threading.Thread(target=process_video).start()

# Function to process video frames and detect objects
def process_video():
    global videoCap, stop_video, target_class
    while videoCap.isOpened() and not stop_video:
        ret, frame = videoCap.read()
        if not ret:
            break

        results = yolo.track(frame, stream=True)
        for result in results:
            classes_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    [x1, y1, x2, y2] = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0])
                    class_name = classes_names[cls]

                if target_class.strip() == "" or target_class.lower() == "all" or target_class.lower() == class_name.lower():
                    colour = get_colours(cls)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        # Convert frame for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((800, 500))
        imgtk = ImageTk.PhotoImage(image=img)
        video_display.imgtk = imgtk
        video_display.configure(image=imgtk)

        if stop_video:
            break
    videoCap.release()

# Function to stop video
def stop_detection():
    global stop_video
    stop_video = True
    video_display.config(image='')

# GUI Setup
root = Tk()
root.title("Video Object Tracker")
root.geometry("1000x700")
root.configure(bg="#cff3ea")  # light blue background

# Header label
header_label = Label(root, text="Real-Time Video Object Detection", bg="#cff3ea",
                     fg="#8b0000", font=("Arial", 18, "bold"))
header_label.pack(pady=10)

# Video selection
video_label = Label(root, text="No video chosen", bg="#fc9551", font=("Arial", 12,"bold"))
video_label.pack(pady=5)
select_button = Button(root, text="Browse Video", bg="#0f5bff", fg="white", font=("Arial", 12, "bold"), command=select_video)
select_button.pack(pady=5)

# Entry for class name
class_label = Label(root, text="Object Class to Detect:", bg="#cff3ea", font=("Arial", 12,"bold"))
class_label.pack(pady=5)
class_entry = Entry(root, font=("Arial", 12), width=25)
class_entry.pack()

# Start and Stop buttons
# Frame to hold the buttons horizontally
button_frame = Frame(root, bg="#cff3ea")
button_frame.pack(pady=20)

# Start and Stop buttons inside the frame
start_button = Button(button_frame, text="Run Detection", bg="#008000", fg="white",
                      font=("Arial", 12, "bold"), command=lambda: set_class_and_start())
start_button.pack(side='left', padx=10,)

stop_button = Button(button_frame, text="Stop Detection", bg="#ff0000", fg="white",
                     font=("Arial", 12, "bold"), command=stop_detection)
stop_button.pack(side='left', padx=10)


# Video display
video_display = Label(root, bg="#cff3ea")
video_display.pack(pady=10)

# Optional progress bar
progress_var = IntVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=800)
progress_bar.pack(pady=5)

# Function to set class and start detection
def set_class_and_start():
    global target_class
    target_class = class_entry.get().strip()
    start_detection()

# Start GUI loop
root.mainloop()
