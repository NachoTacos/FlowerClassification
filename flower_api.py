from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'daisy', 1 : 'dandelion', 2 : 'roses', 3 : 'sunflowers', 4 : 'tulips'}

model = load_model('flowermodel.keras')

model.make_predict_function()
def predict_label(img_path):
    i = image.load_img(img_path, target_size = (180,180))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 180, 180, 3)
    p = (model.predict(i)>0.5).astype("int32")
    class_index = p.argmax()
    return dic[class_index]

#Routes 
@app.route("/", methods =['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods = ['GET','POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        
        img_path =  img.filename
        img.save(img_path)
        
        p = predict_label(img_path)
    return render_template("index.html", prediction = p, img_path = img_path)

if __name__ == '__main__':
    #app.debug = True
    app.run(debug = True)
    
