from flask import Flask
from flask import request
app = Flask(__name__)
from service.GreetingService import GreetingService
from util.ModelHandler import ModelHandler
greetingService = GreetingService()

modelHandler = ModelHandler()

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/servicetest')
def serviceHello():
    return greetingService.sayHello()

@app.route('/paramTest')
def paramHello():
   # param = request.args.get('helloparam')
    modelHandler.init()
    #return param

@app.route('/printExcel')
def readAndPrintExcelFile():
    ### HERE

    ### Read excel file
    ### Print as DF on the screen


    return

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=9999)

