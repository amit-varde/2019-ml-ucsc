# import the Flask class from the flask module
from flask import Flask, render_template
from os import listdir
from os.path import isfile, join
from os import walk
import commands
import subprocess





# create the application object
app = Flask(__name__)

# use decorators to link the function to a url
@app.route('/')
def home():
    return render_template('google.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')  # render a template

@app.route('/nlreco', methods=['GET','POST'])
def nlreco():
   cmd="/Users/amit/ML/ClassProject/2019-ml-ucsc/src/nl-reco.py"
   p1 = subprocess.Popen([cmd], stdout=subprocess.PIPE)
   output = p1.communicate()[0].replace('\n', '<br />')
   return output
    
@app.route('/creco', methods=['GET','POST'])
def creco():
   cmd="/Users/amit/ML/ClassProject/2019-ml-ucsc/src/creco.py "
   p1 = subprocess.Popen([cmd], stdout=subprocess.PIPE)
   output = p1.communicate()[0].replace('\n', '<br />')
   return output

@app.route('/agreco', methods=['GET','POST'])
def agreco():
   cmd="/Users/amit/ML/ClassProject/2019-ml-ucsc/src/nl-reco.py"
   p1 = subprocess.Popen([cmd], stdout=subprocess.PIPE)
   output = p1.communicate()[0].replace('\n', '<br />')
   return output

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
