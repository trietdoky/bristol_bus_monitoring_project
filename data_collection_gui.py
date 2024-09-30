import tkinter as tk
from tkinter import scrolledtext
import subprocess
import threading
import os
import time
from datetime import datetime, timedelta

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Panel")
        
        self.start_button = tk.Button(root, text="START", command=self.start_script)
        self.start_button.pack(pady=5)
        
        
        self.stop_button = tk.Button(root, text="STOP", command=self.stop_script, state=tk.DISABLED)
        self.stop_button.pack(pady=5)
        
        self.log_textbox = scrolledtext.ScrolledText(root, width=50, height=20)
        self.log_textbox.pack(pady=5)
        
        self.process = None
        self.running = False
        self.record_count = 0
        self.start_time = datetime.now()
        self.time_elapsed = 0
        
        self.update_log()

    def start_script(self):
        if self.process is None:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            # self.process = subprocess.run(dc.run_scheduler)
            self.process = subprocess.Popen(['python', 'data_collection_v003.py'],1)
            self.running = True

    def stop_script(self):
        if self.process is not None:
            self.process.terminate()
            self.process = None
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.running = False

    def update_log(self):
        log_file_name = './data/raw/'+str(datetime.now().strftime('%Y%m%d'))+'_'+str(699)+'_log'
        self.record_count += 1
        duration = datetime.now() - self.start_time
        self.time_elapsed = round(duration.total_seconds()/60, 1)
        if os.path.exists(log_file_name):
            with open(log_file_name, "r") as f:
                content = f.read()
                # self.log_textbox.delete(1.0, tk.END)
                self.log_textbox.delete(1.0, tk.END)
                self.log_textbox.insert(tk.END, content)
                self.log_textbox.insert(tk.END, f"Time Elapsed: {self.time_elapsed} minutes,\nNumber of Records obtained {self.record_count}")
        
        self.record_count += 1
        duration = datetime.now() - self.start_time
        self.time_elapsed = round(duration.total_seconds()/60, 1)

        with open('./data/raw/latest_log', "r+") as f:
            content = f.read()
        self.log_textbox.delete(1.0, tk.END)
        self.log_textbox.insert(tk.END, content)
        self.log_textbox.insert(tk.END, f"Time Elapsed: {self.time_elapsed} minutes,\nNumber of Records obtained {self.record_count}")        

        if self.running:
            self.root.after(5000, self.update_log)
        else:
            self.root.after(5000, self.update_log)
        
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()