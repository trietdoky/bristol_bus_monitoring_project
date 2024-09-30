import time
import schedule

# Function to read and print the content of the file
def read_and_print_file_content():
    try:
        with open('random_strings.txt', 'r') as file:
            content = file.read()
            print(f"File content at {time.ctime()}:\n{content}")
    except FileNotFoundError:
        print("File 'random_strings.txt' not found. Waiting for the file to be created...")

# Schedule the read function to run every 3 seconds
schedule.every(3).seconds.do(read_and_print_file_content)

# Run the scheduled tasks
if __name__ == '__main__':
    while True:
        schedule.run_pending()
        time.sleep(1)