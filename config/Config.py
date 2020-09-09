from os import environ
import yaml
from yaml import safe_load
from sqlalchemy import create_engine
import pathlib

class Config:
    """Set Flask configuration vars from .env file."""

    # Load Config.yml--------------------------------------------
    fn = pathlib.Path(__file__).parent / '../config.yml'
    with open(fn, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    # -----------------------------------------------------------

    # Local Config -------------------------------------------
    L_SERVER = config['local-database']['server']
    L_DATABASE = config['local-database']['database']
    L_DRIVER = config['local-database']['driver']
    L_USERNAME = config['local-database']['username']
    L_PASSWORD = config['local-database']['password']
    L_DATABASE_CONNECTION = config['local-database']['connection-string']

    # Profleet Config ---------------------------------------------
    PF_SERVER = config['profleet']['server']
    PF_DATABASE = config['profleet']['database']
    PF_DRIVER = config['profleet']['driver']
    PF_USERNAME = config['profleet']['username']
    PF_PASSWORD = config['profleet']['password']
    PF_DATABASE_CONNECTION = config['profleet']['connection-string']

    # icdb Config -------------------------------------------
    PP_SERVER = config['icdb']['server']
    PP_DATABASE = config['icdb']['database']
    PP_DRIVER = config['icdb']['driver']
    PP_USERNAME = config['icdb']['username']
    PP_PASSWORD = config['icdb']['password']
    PP_DATABASE_CONNECTION = config['icdb']['connection-string']

    # CRM Customer Config ---------------------------------------
    CRM_SERVER = config['crm']['server']
    CRM_DATABASE = config['crm']['database']
    CRM_DRIVER = config['crm']['driver']
    CRM_USERNAME = config['crm']['username']
    CRM_PASSWORD = config['crm']['password']
    CRM_DATABASE_CONNECTION = config['crm']['connection-string']

    # Credential
    APP_CREDENTIAL = '9GLQB5MGB3CXsBkDw7uzZfiY3oCDbuO5'

    local = False

    if(local):
        l_engine = create_engine(L_DATABASE_CONNECTION)
        l_connection = l_engine.connect()
    else:
        pp_engine = create_engine(PP_DATABASE_CONNECTION)
        pp_connection = pp_engine.connect()
    # -----------------------------------------------------------

