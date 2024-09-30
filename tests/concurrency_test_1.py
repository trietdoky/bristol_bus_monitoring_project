import time
import random
import string
import schedule

# Function to generate a random 10-character string
def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits  # Characters include both letters and digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# Function to write the random string to a text file
def write_random_string_to_file():
    random_string = generate_random_string()
    with open('random_strings.txt', 'w') as file:
        file.write(random_string)  # Append the string with a newline
    print(f"Written string: {random_string}")

# Schedule the write function to run every 5 seconds
schedule.every(5).seconds.do(write_random_string_to_file)

# Run the scheduled tasks
if __name__ == '__main__':
    while True:
        schedule.run_pending()
        time.sleep(1)