import requests

# Set the API endpoint
url = "http://localhost:5000/predict"

# Path to the image you want to test
image_path = "Rojo.jpeg"

# Open the image file in binary mode
with open(image_path, "rb") as img_file:
    # Create the payload with the image file
    files = {'file': img_file}

    # Send the POST request to the API
    response = requests.post(url, files=files)

# Print out the response from the API
print("Response Status Code:", response.status_code)
print("Prediction Result:", response.json())
