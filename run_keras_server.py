# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from flask import Flask
from flask import render_template, request, send_from_directory

import time
import dog_breed_application

app = Flask(__name__)

# webpage 

@app.route('/master')
def main():
    return render_template('master.html')

# web page for image upload https://sodocumentation.net/de/flask/topic/5459/datei-uploads
@app.route('/go', methods=['POST'])

def go():
    
    if request.method == 'POST':
      try:
        img_file = request.files['img']
        img_file.seek(0)
        img_file.save('upload.jpg')
        dog_breed_application.dog_breed_prediction('upload.jpg')
        result = "<img src='result.png'>"
      except :
        result = "<b>No valid image found</b>" 
        
    else :
      result = "<b>Please select an image</b>"
          
    result = "<img class='NO-CACHE'  src='result.png?%s'>" % int(time.time())
    return render_template( 'go.html', query=result )

# Display image
@app.route('/result.png')
def image():
    return send_from_directory('.', 'result.png')

def main():
    app.run(host='2.6.2.6', port=3001, debug=False, threaded=False)

if __name__ == '__main__':
    main()