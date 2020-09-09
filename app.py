from flask import Flask
from flask import request
from flask_sqlalchemy import SQLAlchemy
from service.GreetingService import GreetingService
from util.ModelHandler import ModelHandler
#from .config.Config import Config

app = Flask(__name__)

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


# return param

@app.route('/printExcel')
def readAndPrintExcelFile():
    ### HERE

    return greetingService.read()

    ### Read excel file
    ### Print as DF on the screen


if __name__ == '__main__':
    app.secret_key = '_5#y2L"F4Q8z]/'
    app.debug = True
    app.run(host='0.0.0.0', port=9999)
