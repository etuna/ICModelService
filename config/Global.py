
from ..model.Base import session_factory
import pandas as pd
from ..service import ServerLogService, ApplicationLogService, CriteriaService, CustomerService, ProductService, AuthenticationService
from app import application
import logging
from flask import request
from ..util import MarkupHandler, TumiHandler
from .Config import Config


db_session = session_factory()
serverLogService = ServerLogService.ServerLogService(db_session)
applicationLogService = ApplicationLogService.ApplicationLogService(db_session)
criteriaService = CriteriaService.CriteriaService(db_session)
markupHandler = MarkupHandler.MarkupHandler(db_session)
tumiHandler = TumiHandler.TumiHandler(db_session)
customerService = CustomerService.CustomerService(db_session)
productService = ProductService.ProductService(db_session)
authenticationService = AuthenticationService.AuthenticationService(db_session)
customers = pd.DataFrame(customerService.getAll())
products = pd.DataFrame(productService.getAll())
customers = customers.drop_duplicates()
products = products.drop_duplicates()
credential = Config.APP_CREDENTIAL
competitors = ['Continental', 'Goodyear', 'Pirelli', 'Michelin', 'Hankook', 'Petlas']
