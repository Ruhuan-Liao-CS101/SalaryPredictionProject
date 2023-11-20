import flask
from flask import Flask, render_template, request

app=Flask(__name__)

def init():
    print("initializing... ") 
  
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    tagValuePairs = request.form.to_dict()
    print(tagValuePairs)

    return render_template('result.html', rc="hello from predict..")
 


if __name__ == '__main__':
    init()
    app.run(debug=True, port=9090)
   
