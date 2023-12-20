from flask import Flask, render_template, request, jsonify
from azure.storage.blob import BlobServiceClient
from PIL import Image
from io import BytesIO
from ultralytics import YOLO


app = Flask(__name__)

# Your Azure Blob Storage connection string
CONNECTION_STRING = "AA"#replace with your azure key
CONTAINER_NAME = "images"  # Name of the container for image uploads
PROCESSED_CONTAINER_NAME = "processedimage"  # Name of the container for processed images

# Initialize the YOLO model
model = None

# Add more models as needed

@app.route('/', methods=['GET', 'POST'])
def index():
    processed_link = None  # Initialize processed_link variable
     # Initialize the model variable
    
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        user_input = request.form.get('user_input')  # Get user input from the form

        # Upload image to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(imagefile.filename)
        blob_client.upload_blob(imagefile.read(), overwrite=True)

        # Select the appropriate model based on user input
        if user_input == 'lung':
            model = YOLO('pneunomia.pt')
        elif user_input == 'eye':
            model = YOLO('eye.pt')
        elif user_input =='intestine':
            model = YOLO('intestine.pt')
        # Add more scenarios here
        
        if model:
            processed_image_data = process_image(imagefile.filename, model)
        else:
            processed_image_data = b''  # Default case, no processing

        # Upload the processed result image back to Azure Blob Storage
        processed_image_filename = f"processed_{user_input}_{imagefile.filename}"
        processed_blob_client = blob_service_client.get_blob_client(
            PROCESSED_CONTAINER_NAME, processed_image_filename
        )
        processed_blob_client.upload_blob(processed_image_data, overwrite=True)

        # Generate the link to the processed image
        processed_link = f"https://pneumoniaimage.blob.core.windows.net/{PROCESSED_CONTAINER_NAME}/{processed_image_filename}"
        
        return render_template('index.html', processed_link=processed_link)

    return render_template('index.html')

# Rest of your code remains unchanged
@app.route('/get_processed_link', methods=['GET'])
def get_processed_link():
    processed_link = request.args.get('processed_link')
    return jsonify({'processed_link': processed_link})


def process_image(image_filename, model):
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blob_client = container_client.get_blob_client(image_filename)
    image_data = blob_client.download_blob().readall()
    original_image = Image.open(BytesIO(image_data))
    original_image.save("original.jpg", format='JPEG')
    pil_image = Image.open(BytesIO(image_data))
    
    # Process image with YOLO model
    results = model(pil_image)  
    
    # Convert results to an image
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

    # Save processed image as bytes
    processed_image_buffer = BytesIO()
    im.save(processed_image_buffer, format='JPEG')  # Change 'JPEG' to 'PNG' if you want PNG format
    processed_image_data = processed_image_buffer.getvalue()

    return processed_image_data

if __name__ == '__main__':
    app.run(port=3000, debug=True)