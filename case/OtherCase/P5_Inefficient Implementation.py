import base64
import easyocr
import io
from PIL import Image

# Demo base64 string of an example image
# This is a placeholder. Replace 'base64_image_string' with an actual base64-encoded image string for real testing.
base64_image_string = (
    "iVBORw0KGgoAAAANSUhEUgAAAKAAAABwCAYAAACpJ55tAAAF0UlEQVR42u3SAREAAAjDIZz/"
    "eLjEhFeDRAONHgFFHklIVokiWkliWkmGVUiGVkmGVkgGVklGVUlGVkmGVklmXVklWXkmZWkmG"
    "VWkmGV0eQ+A3wAAf9AAAAAAXfAArHC9xhPgAAAAElFTkSuQmCC"
)

# Function to decode the base64 image and convert it to a format usable by EasyOCR
def decode_base64_image(encoded_str):
    image_bytes = base64.b64decode(encoded_str)
    return Image.open(io.BytesIO(image_bytes))

# Create a method that uses EasyOCR's Reader to recognize text
def extract_text_from_image(base64_encoded_image, keywords):
    reader = easyocr.Reader(["en"])
    # Decode the base64-encoded image
    image = decode_base64_image(base64_encoded_image)
    
    # Use EasyOCR to extract text data from the image
    text_results = reader.readtext(np.array(image), detail=1, paragraph=False)
    
    # Parse the results to extract text and their positions
    found_keywords = {}
    for result in text_results:
        detected_text = result[1]  # Result text
        for keyword in keywords:  # Check if keyword matches
            if keyword in detected_text:
                found_keywords[keyword] = (
   adjal***Looking, Developer simulated re substituitional adjustment"

Advantages